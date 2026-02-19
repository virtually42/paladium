# Paladium: Automatic Differentiation in Scala 3

Paladium is a minimal automatic differentiation library inspired by [micrograd](https://github.com/karpathy/micrograd). It implements reverse-mode autodiff (backpropagation) using a clean, layered architecture.

## Design Decisions

### Layered Architecture

Rather than putting everything in one place, we separate concerns into three layers:

```
┌─────────────────────────────────────────┐
│  AutoGrad   (convenience layer)         │  ← "Give me value AND gradients"
│  expr.trace → Traced(value, grads)      │
├─────────────────────────────────────────┤
│  Grad       (differentiation)           │  ← "Compute gradients"
│  Grad.backward(expr) → Map[String, A]   │
├─────────────────────────────────────────┤
│  Value      (expression tree)           │  ← "Build & evaluate expressions"
│  x + y, x * y, x ** y, etc.             │
└─────────────────────────────────────────┘
```

**Why this separation?**

1. **Value** is a pure expression tree — it only knows how to represent and evaluate math
2. **Grad** is an interpreter that walks the tree and computes derivatives
3. **AutoGrad** combines both for the common "forward + backward" pattern

This makes each piece easier to understand, test, and extend. You could add other interpreters (compile to CUDA, symbolic simplification) without touching Value.

---

## Layer 1: Value — Expression Trees

### The Core Idea

Instead of computing `2 * 3 + 2` immediately, we build a tree that *represents* the computation:

```scala
val x = Value(2.0)
val y = Value(3.0)
val expr = x * y + x    // doesn't compute yet — builds a tree
expr.eval               // NOW it computes: 8.0
```

### What the Tree Looks Like

When you write `x * y + x`, you're building this structure:

```
        Add
       /   \
      /     \
    Mul      Lit(2.0)
   /   \
  /     \
Lit(2.0) Lit(3.0)
```

Each node is a case of the `Value` enum:

```scala
enum Value[A]:
  case Var(id: String, data: A)   // named variable (for gradients)
  case Lit(data: A)               // literal value
  case Const(n: Int)              // integer constant
  case Add(left: Value[A], right: Value[A])
  case Sub(left: Value[A], right: Value[A])
  case Mul(left: Value[A], right: Value[A])
  case Div(left: Value[A], right: Value[A])
  case Pow(base: Value[A], exp: Value[A])
  case Neg(value: Value[A])
```

### Understanding Const vs Lit vs Var

You might wonder: why is `Const` limited to `Int` while `Lit` holds any type `A`? This reflects a fundamental distinction in mathematical notation.

**Structural constants (`Const`)**: The integers 0, 1, 2, 3... that appear in the *form* of mathematical expressions:

- `2x + 1` — the 2 and 1 are structural
- `x²` — the exponent 2 is structural
- `3x² - 4x + 5` — the coefficients 3, 4, 5 are structural

These are **exact**, **type-independent**, and part of the expression's shape. Whether you later evaluate as `Double`, `Float`, or `BigDecimal`, the formula `3x² - 4x + 5` has the same structure. The conversion to your numeric type happens at evaluation time via `NumberLike.fromInt`.

**Literal values (`Lit`)**: Concrete values already in your target type. Use these when you have actual numeric data:

```scala
val measured = Value(3.14159)  // Lit(3.14159) — a Double you computed or measured
```

**Named variables (`Var`)**: Values you want to track for gradient computation:

```scala
val x = Value.variable("x", 2.0)  // Var("x", 2.0) — gradients will include "x"
```

**Why not `Const(n: Double)`?**

In mathematics, you never write `3.14159 * r²` — you write `π * r²`. Named mathematical constants (π, e, √2) are kept symbolic, not approximated. A floating-point "constant" embedded in an expression is:

- Imprecise — which `3.14159`? Truncated? Rounded?
- Unrecognizable — symbolic simplification can't identify it as π
- Not idiomatic — it mixes exact symbolic structure with inexact numerics

If you need π or e, the proper approach is to add named constants:

```scala
// Future extension
case object Pi extends Value[A]
case object E extends Value[A]
```

Or treat them as parameters via `Var("pi", 3.14159265...)`.

**Summary**:

| Type | Purpose | Example |
|------|---------|---------|
| `Const(n: Int)` | Structural constants in formulas | `2` in `2x + 1` |
| `Lit(data: A)` | Concrete values in your numeric type | Measured data, computed results |
| `Var(id, data)` | Named values for gradient tracking | `x` in `∂f/∂x` |

### Evaluation: Walking the Tree

To get a number out, we recursively evaluate:

```scala
def eval(using num: NumberLike[A]): A =
  this match
    case Lit(data)        => data                              // just return it
    case Add(left, right) => num.plus(left.eval, right.eval)   // recurse both sides
    case Mul(left, right) => num.times(left.eval, right.eval)
    // ... etc
```

For our tree `(2 * 3) + 2`:

```
        Add                         Add
       /   \                       /   \
     Mul    Lit(2.0)    →       6.0    2.0    →    8.0
    /   \
  2.0   3.0
```

### Why Trees Matter

The tree preserves the *structure* of computation. This is crucial because:
- We need to know HOW a result was computed to find derivatives
- `x * x` and `x + x` both use `x` twice, but have different derivatives

---

## Layer 2: Grad — Computing Derivatives

### The Goal

Given an expression like `f(x, y) = x * y + x`, find:
- ∂f/∂x (how much does f change when x changes?)
- ∂f/∂y (how much does f change when y changes?)

### Named Variables

To track which inputs we want gradients for, we use `Var`:

```scala
val x = Value.variable("x", 2.0)   // Var("x", 2.0)
val y = Value.variable("y", 3.0)   // Var("y", 3.0)
val expr = x * y + x
```

Now our tree looks like:

```
          Add
         /   \
        /     \
      Mul     Var("x", 2.0)
     /   \
    /     \
Var("x",2) Var("y",3)
```

### The Chain Rule

Derivatives flow *backwards* through the tree. This is the key insight of backpropagation.

For each operation, we know the local derivative rules:

| Operation | ∂/∂left | ∂/∂right |
|-----------|---------|----------|
| `left + right` | 1 | 1 |
| `left - right` | 1 | -1 |
| `left * right` | right | left |
| `left / right` | 1/right | -left/right² |
| `base ** exp` | exp × base^(exp-1) | base^exp × ln(base) |

The **chain rule** says: multiply the "upstream" gradient by the local derivative.

### Backward Pass: A Worked Example

Let's trace through `f(x,y) = x * y + x` at x=2, y=3:

**Step 1: Forward pass (build tree, compute values)**

```
              Add [8.0]
             /         \
            /           \
      Mul [6.0]      Var("x") [2.0]
       /      \
      /        \
Var("x")[2.0]  Var("y")[3.0]
```

**Step 2: Backward pass (propagate gradients)**

Start at the output with gradient = 1 (we want ∂f/∂f = 1):

```
              Add [8.0]
              upstream = 1.0
             /         \
            /           \
      Mul [6.0]      Var("x") [2.0]
```

**At the Add node:** Both children get the upstream gradient unchanged (since ∂(a+b)/∂a = 1)

```
              Add
              ↓ upstream = 1.0
         ┌────┴────┐
         ↓         ↓
        1.0       1.0
         ↓         ↓
      Mul       Var("x")
                  │
                  └→ accumulate: x gets +1.0
```

**At the Mul node:** Left child gets `upstream × right.eval`, right gets `upstream × left.eval`

```
      Mul
      ↓ upstream = 1.0
   ┌──┴──┐
   ↓     ↓
  y=3   x=2     (local derivatives)
   ↓     ↓
  3.0   2.0     (upstream × local)
   ↓     ↓
Var("x") Var("y")
   │       │
   └→ +3.0 └→ +2.0
```

**Final gradient accumulation:**

```
x: 1.0 (from Add's right) + 3.0 (from Mul's left) = 4.0
y: 2.0 (from Mul's right) = 2.0
```

Verify: f(x,y) = xy + x, so ∂f/∂x = y + 1 = 3 + 1 = 4 ✓ and ∂f/∂y = x = 2 ✓

### The Implementation

```scala
object Grad:
  def backward[A: NumberLike](expr: Value[A]): Map[String, A] =
    backwardAccum(expr, num.fromInt(1), Map.empty)  // start with upstream=1

  private def backwardAccum[A](
    expr: Value[A],
    upstream: A,           // gradient flowing down from parent
    accum: Map[String, A]  // accumulated gradients so far
  )(using num: NumberLike[A]): Map[String, A] =
    expr match
      case Var(id, _) =>
        // Reached a variable: add upstream to its gradient
        val existing = accum.getOrElse(id, num.fromInt(0))
        accum.updated(id, num.plus(existing, upstream))

      case Lit(_) | Const(_) =>
        accum  // constants have no gradient to propagate

      case Add(l, r) =>
        // Both children get upstream unchanged
        val a1 = backwardAccum(l, upstream, accum)
        backwardAccum(r, upstream, a1)

      case Mul(l, r) =>
        // left gets upstream * right.eval, right gets upstream * left.eval
        val a1 = backwardAccum(l, num.times(upstream, r.eval), accum)
        backwardAccum(r, num.times(upstream, l.eval), a1)

      // ... other cases follow the same pattern
```

### Gradient Accumulation: Why It Matters

When a variable appears multiple times, its gradients are **summed**:

```scala
val x = Value.variable("x", 3.0)
val expr = x * x   // x appears twice!
```

Tree structure:

```
       Mul
      /   \
     /     \
  Var("x") Var("x")   ← same variable, two references
```

Backward pass:
- Left child: upstream(1) × right.eval(3) = 3 → x gets +3
- Right child: upstream(1) × left.eval(3) = 3 → x gets +3
- Total: x gradient = 6

This matches calculus: d/dx(x²) = 2x = 6 at x=3 ✓

---

## Layer 3: AutoGrad — Putting It Together

### The Common Pattern

In machine learning, you almost always want both:
1. The output value (forward pass)
2. The gradients (backward pass)

AutoGrad bundles these:

```scala
case class Traced[A](value: A, grads: Map[String, A])

object AutoGrad:
  def trace[A: NumberLike](expr: Value[A]): Traced[A] =
    Traced(expr.eval, Grad.backward(expr))
```

### Usage

```scala
import AutoGrad.syntax.*

val x = Value.variable("x", 2.0)
val loss = x * x + x

val result = loss.trace
// Traced(
//   value = 6.0,          // 2² + 2 = 6
//   grads = Map("x" -> 5.0)  // 2x + 1 = 5
// )

result.value      // 6.0
result.grads("x") // 5.0
```

### Neural Network Training Loop (Conceptual)

```scala
// Pseudocode for how this would be used
for epoch <- 1 to 1000 do
  val Traced(loss, grads) = model.forward(input).trace

  // Update weights: w = w - learningRate * grad
  for (name, grad) <- grads do
    weights(name) -= learningRate * grad
```

---

## Complete Example: Polynomial Gradient

Let's trace through f(x) = 3x² - 4x + 5 at x = 2:

```scala
val x = Value.variable("x", 2.0)
val expr = Value.const(3) * (x ** Value.const(2)) - Value.const(4) * x + Value.const(5)

val result = AutoGrad.trace(expr)
// value = 3(4) - 4(2) + 5 = 12 - 8 + 5 = 9
// grad  = 6x - 4 = 12 - 4 = 8
```

The computation graph:

```
                            Add [9.0]
                           /         \
                          /           \
                    Sub [7.0]       Const(5) [5.0]
                   /         \
                  /           \
           Mul [12.0]      Mul [8.0]
           /      \        /      \
          /        \      /        \
     Const(3)    Pow   Const(4)  Var("x")
       [3.0]    [4.0]   [4.0]     [2.0]
                /    \
               /      \
          Var("x")  Const(2)
            [2.0]     [2.0]
```

Backward pass (simplified):
1. Start at Add with upstream = 1
2. Sub gets 1, Const(5) gets 1 (no var, ignored)
3. At Sub: left Mul gets 1, right Mul gets -1
4. Left Mul (3 × x²): Pow gets 1 × 3 = 3
5. At Pow: x gets 3 × 2 × 2^(2-1) = 3 × 2 × 2 = 12
6. Right Mul (4 × x): x gets -1 × 4 = -4
7. Total for x: 12 + (-4) = 8 ✓

---

## API Reference

### Value — Expression Building

```scala
// Create values
Value(2.0)                    // Lit(2.0)
Value.const[Double](3)        // Const(3) — converted via NumberLike
Value.variable("x", 2.0)      // Var("x", 2.0) — tracked for gradients

// Operations (return new Value nodes)
x + y    // Add
x - y    // Sub
x * y    // Mul
x / y    // Div
x ** y   // Pow (also: x.pow(y))
-x       // Neg

// Evaluate
expr.eval   // requires NumberLike[A] in scope
```

### Grad — Differentiation

```scala
// Compute gradients for all variables
Grad.backward(expr)   // Map[String, A]

// With syntax extension
import Grad.syntax.*
expr.backward         // Map[String, A]
```

### AutoGrad — Combined Forward + Backward

```scala
// Get value and gradients together
AutoGrad.trace(expr)   // Traced[A](value, grads)

// With syntax extension
import AutoGrad.syntax.*
expr.trace             // Traced[A]

// Destructuring
val Traced(value, grads) = expr.trace
```

---

## Extending with New Number Types

Implement `NumberLike` to support new numeric types:

```scala
given NumberLike[BigDecimal] with
  def plus(x: BigDecimal, y: BigDecimal) = x + y
  def minus(x: BigDecimal, y: BigDecimal) = x - y
  def times(x: BigDecimal, y: BigDecimal) = x * y
  def div(x: BigDecimal, y: BigDecimal) = x / y
  def pow(x: BigDecimal, exp: BigDecimal) = x.pow(exp.toInt)  // simplified
  def log(x: BigDecimal) = BigDecimal(math.log(x.toDouble))
  def fromInt(n: Int) = BigDecimal(n)
```

Then all three layers work automatically with your type.

---

## DSL Extension — Concise Syntax

The `Dsl` module provides ergonomic extensions for building expressions with less boilerplate.

### Setup

```scala
import paladium.*
import paladium.Dsl.{*, given}
```

### Automatic Variable Naming with `^` or `toVar`

Instead of manually specifying variable names:

```scala
// Without DSL
val x = Value.variable("x", 2.0)
val y = Value.variable("y", 3.0)

// With DSL — variable name captured automatically from val binding
val x = 2.0.^      // Value.Var("x", 2.0)
val y = 3.0.toVar  // Value.Var("y", 3.0)
```

The `^` operator symbolizes "lifting" a value into the expression DSL. Use `toVar` if you prefer explicit naming.

### Implicit Number Promotion

Raw numbers are automatically promoted to `Value.Lit`:

```scala
// Without DSL
val expr = x * Value(2.0) + Value(1.0)

// With DSL — numbers auto-promote
val expr = x * 2.0 + 1.0
```

### Complete Example

```scala
import paladium.*
import paladium.Dsl.{*, given}
import paladium.AutoGrad.syntax.*

val x = 2.0.^
val y = 3.0.^

// Clean expression syntax — no Value() wrappers needed
val expr = x * x + 2.0 * y

val Traced(value, grads) = expr.trace
// value = 4 + 6 = 10
// grads = Map("x" -> 4.0, "y" -> 2.0)
```

### How It Works

The DSL uses two Scala features:

1. **sourcecode library** — `sourcecode.Name` captures the name of the `val` binding at compile time
2. **Scala 3 Conversion** — `given Conversion[A, Value[A]]` allows implicit promotion of `A` to `Value.Lit(a)`

Both extensions require a `NumberLike[A]` instance, so they work with any numeric type you've implemented support for.

---

## SymbolicGrad — Building Gradient Expressions

### The Problem with Numerical Gradients

The `Grad` module computes numerical gradient values immediately:

```scala
val x = Value.variable("x", 2.0)
val expr = x * x
Grad.backward(expr)  // Map("x" -> 4.0) — a concrete number
```

This works well for CPU execution, but has limitations:

1. **GPU/CUDA execution**: GPUs are efficient when you batch operations. Computing gradients one-by-one defeats the purpose.
2. **Graph optimization**: You can't simplify or fuse operations on numbers that are already computed.
3. **Code generation**: You can't emit optimized code from a number.
4. **Inspection**: You can't see *how* the gradient is computed, only the final value.

### The Solution: Symbolic Gradients

`SymbolicGrad` builds gradient *expressions* instead of computing values:

```scala
import SymbolicGrad.syntax.*

val x = Value.variable("x", 2.0)
val expr = x * x

// Numerical: gives you a number
Grad.backward(expr)           // Map("x" -> 4.0)

// Symbolic: gives you an expression tree
SymbolicGrad.backward(expr)   // Map("x" -> Add(Const(0), Add(Mul(Const(1), Var("x")), Mul(Const(1), Var("x")))))
```

The symbolic gradient for `x` is an expression that, when evaluated, gives 4.0:

```scala
val gradExpr = SymbolicGrad.backward(expr)("x")
gradExpr.eval  // 4.0
```

### How It Works

The key difference is in what flows through the backward pass:

| Aspect | `Grad` | `SymbolicGrad` |
|--------|--------|----------------|
| Upstream type | `A` (a number) | `Value[A]` (an expression) |
| Accumulator | `Map[String, A]` | `Map[String, Value[A]]` |
| Multiplication | `num.times(upstream, r.eval)` | `Mul(upstream, r)` |
| Result | Computed values | Expression trees |

Compare the `Mul` case in both implementations:

```scala
// Grad — computes immediately
case Mul(l, r) =>
  val a1 = backwardAccum(l, num.times(upstream, r.eval), accum)  // upstream * 3.0
  backwardAccum(r, num.times(upstream, l.eval), a1)              // upstream * 2.0

// SymbolicGrad — builds expressions
case Mul(l, r) =>
  val a1 = backwardAccum(l, Mul(upstream, r), accum)  // Mul(upstream, Var("y"))
  backwardAccum(r, Mul(upstream, l), a1)              // Mul(upstream, Var("x"))
```

### Why This Matters

With symbolic gradients, you have two expression trees:

```
Forward Expression              Backward Expressions
─────────────────              ────────────────────
     Mul                       x: Mul(Const(1), Var("y"))
    /   \                      y: Mul(Const(1), Var("x"))
Var(x)  Var(y)
```

Both use the same `Value[A]` ADT, which means:

1. **Same interpreter for both**: One `eval` implementation handles forward and backward
2. **Optimization passes**: Simplify `Mul(Const(1), x)` → `x` works on both
3. **Visualization**: Render both graphs with the same Mermaid/DOT generator
4. **GPU compilation**: Lower both to CUDA kernels using the same compiler

### The Interpreter Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Interpreters                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │    eval     │  │  toMermaid  │  │    toCuda       │  │
│  │  → A        │  │  → String   │  │  → CudaKernel   │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Forward:   Value[A]  ──────────────────────────────►   │
│                                                         │
│  Backward:  Map[String, Value[A]]  ─────────────────►   │
│             (from SymbolicGrad)                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### API Reference

```scala
// Compute symbolic gradients
SymbolicGrad.backward(expr)   // Map[String, Value[A]]

// With syntax extension
import SymbolicGrad.syntax.*
expr.symbolicBackward         // Map[String, Value[A]]

// Evaluate a gradient expression
val gradExprs = expr.symbolicBackward
gradExprs("x").eval           // A — the numerical gradient value
```

### Example: Inspecting Gradient Structure

```scala
import paladium.*
import SymbolicGrad.syntax.*

val x = Value.variable("x", 2.0)
val y = Value.variable("y", 3.0)
val expr = x * y

val grads = expr.symbolicBackward

// The gradient for x is symbolically: upstream * y
// Where upstream starts as Const(1)
grads("x") match
  case Value.Add(Value.Const(0), Value.Mul(Value.Const(1), Value.Var("y", _))) =>
    println("Gradient for x is: 1 * y")
```

### Symbolic vs Numerical: When to Use Which

| Use Case | Use `Grad` | Use `SymbolicGrad` |
|----------|------------|-------------------|
| Quick gradient check | ✓ | |
| CPU training loop | ✓ | |
| GPU/TPU execution | | ✓ |
| Graph visualization | | ✓ |
| Expression simplification | | ✓ |
| Code generation | | ✓ |
| Debugging gradient flow | | ✓ |

---

## The Chain Rule — An Intuitive Guide

### Why the Chain Rule?

The chain rule is the mathematical foundation of backpropagation. If you understand it intuitively, automatic differentiation becomes transparent.

### Leibniz's Notation: The Fraction Intuition

Leibniz's notation writes derivatives as fractions: dy/dx means "the change in y per unit change in x."

#### The Basic Idea

Imagine you're converting currencies:

```
100 USD → ? EUR → ? GBP
```

If 1 USD = 0.85 EUR and 1 EUR = 0.88 GBP, then:

```
dEUR/dUSD = 0.85    (euros per dollar)
dGBP/dEUR = 0.88    (pounds per euro)

dGBP/dUSD = dGBP/dEUR × dEUR/dUSD = 0.88 × 0.85 = 0.748
```

The chain rule says: **rates multiply**. If y changes 3× as fast as x, and z changes 2× as fast as y, then z changes 6× as fast as x.

#### The Chain Rule Formula

For a composition y = f(g(x)), where u = g(x) is the intermediate:

```
dy    dy   du
── = ── × ──
dx    du   dx
```

Read it as: "the rate of y with respect to x equals the rate of y with respect to u, times the rate of u with respect to x."

The notation makes it look like fractions canceling — and while that's not *quite* what's happening mathematically, it's an excellent mental model.

#### Worked Example: f(x) = (x²)³

Let's find df/dx where f(x) = (x²)³.

**Step 1: Identify the chain**

```
x  →  u = x²  →  f = u³
```

**Step 2: Find each derivative**

```
du
── = 2x        (derivative of x²)
dx

df
── = 3u²       (derivative of u³)
du
```

**Step 3: Apply the chain rule**

```
df   df   du
── = ── × ──
dx   du   dx

df
── = 3u² × 2x
dx

   = 3(x²)² × 2x    (substitute u = x²)

   = 6x⁵
```

**Verify**: f(x) = (x²)³ = x⁶, and d/dx(x⁶) = 6x⁵ ✓

#### Multiple Variables: Partial Derivatives

For functions of multiple variables, we use partial derivatives. If z = f(x, y):

```
∂z
──    means "rate of change of z as x varies, holding y constant"
∂x
```

For a chain like z = f(u, v) where u = g(x, y) and v = h(x, y):

```
∂z   ∂z   ∂u     ∂z   ∂v
── = ── × ──  +  ── × ──
∂x   ∂u   ∂x     ∂v   ∂x
```

This is the **multivariate chain rule**: sum over all paths from z to x.

#### Backpropagation in Leibniz Notation

Consider our earlier example: f(x, y) = x·y + x

```
        f = a + b
           /     \
      a = x·y    b = x
```

Using Leibniz notation, for each variable we sum over all paths:

**Path analysis for x:**

```
Path 1: f → a → x
        ∂f   ∂a
        ── × ── = 1 × y = y
        ∂a   ∂x

Path 2: f → b → x
        ∂f   ∂b
        ── × ── = 1 × 1 = 1
        ∂b   ∂x

Total:  ∂f/∂x = y + 1
```

**Path analysis for y:**

```
Path 1: f → a → y
        ∂f   ∂a
        ── × ── = 1 × x = x
        ∂a   ∂y

Total:  ∂f/∂y = x
```

At x=2, y=3: ∂f/∂x = 3 + 1 = 4, ∂f/∂y = 2. This matches our earlier calculation.

#### The "Upstream Gradient" Intuition

In backpropagation, we propagate an "upstream gradient" from output to inputs. In Leibniz notation:

```
upstream = ∂Loss/∂(current node)
```

At each node, we multiply:

```
∂Loss       ∂Loss        ∂(current)
─────────── = ─────────── × ──────────
∂(child)     ∂(current)     ∂(child)
             ↑               ↑
          upstream       local derivative
```

This is why the code does `backwardAccum(child, upstream * localDerivative, accum)`.

---

## The Chain Rule — Formal Treatment

### Lagrange's Notation

Lagrange's notation uses primes for derivatives: f'(x) denotes the derivative of f at x.

```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

For higher derivatives: f''(x), f'''(x), or f⁽ⁿ⁾(x) for the nth derivative.

### The Chain Rule (Single Variable)

**Theorem**: Let f and g be differentiable functions. If h(x) = f(g(x)), then:

```
h'(x) = f'(g(x)) · g'(x)
```

**Proof sketch**: Using the limit definition:

```
h'(x) = lim[Δx→0] (h(x+Δx) - h(x)) / Δx

      = lim[Δx→0] (f(g(x+Δx)) - f(g(x))) / Δx
```

Let Δu = g(x+Δx) - g(x). Then g(x+Δx) = g(x) + Δu, and:

```
      = lim[Δx→0] (f(g(x) + Δu) - f(g(x))) / Δx

      = lim[Δx→0] [(f(g(x) + Δu) - f(g(x))) / Δu] · [Δu / Δx]

      = f'(g(x)) · g'(x)
```

### The Multivariable Chain Rule

**Theorem**: Let f: ℝⁿ → ℝ be differentiable, and let g₁, ..., gₙ: ℝᵐ → ℝ be differentiable. Define h: ℝᵐ → ℝ by:

```
h(x₁, ..., xₘ) = f(g₁(x₁, ..., xₘ), ..., gₙ(x₁, ..., xₘ))
```

Then for each variable xⱼ:

```
∂h        n    ∂f     ∂gᵢ
───── =  Σ   ───── · ─────
∂xⱼ     i=1  ∂gᵢ     ∂xⱼ
```

In vector notation with the gradient ∇f = (∂f/∂g₁, ..., ∂f/∂gₙ):

```
∂h
───── = ∇f · (∂g₁/∂xⱼ, ..., ∂gₙ/∂xⱼ)
∂xⱼ
```

### Formal Definition of Reverse-Mode Autodiff

Let G = (V, E) be a directed acyclic graph representing a computation, where:
- V = {v₁, ..., vₙ} are computational nodes
- E ⊆ V × V are directed edges (vᵢ, vⱼ) meaning vⱼ depends on vᵢ
- vₙ is the output node (loss function)

For each node vᵢ, define the **adjoint**:

```
v̄ᵢ = ∂vₙ/∂vᵢ
```

The adjoint represents "how much the output changes per unit change in this node."

**Reverse-mode algorithm**:

1. **Initialize**: v̄ₙ = 1 (the output's derivative with respect to itself)

2. **Backward pass**: Process nodes in reverse topological order. For each node vᵢ with children C(vᵢ):

```
v̄ᵢ = Σ[vⱼ ∈ C(vᵢ)] v̄ⱼ · (∂vⱼ/∂vᵢ)
```

3. **Result**: For input variables x₁, ..., xₖ, their adjoints x̄₁, ..., x̄ₖ are the desired gradients.

### Complexity Analysis

For a computation with:
- n operations (nodes)
- m inputs (variables)

| Method | Time Complexity | Space Complexity |
|--------|-----------------|------------------|
| Symbolic differentiation | O(n) per input = O(nm) | O(n) per gradient |
| Numerical (finite diff) | O(n) per input = O(nm) | O(1) |
| Forward-mode autodiff | O(n) per input = O(nm) | O(n) |
| **Reverse-mode autodiff** | **O(n) total** | O(n) |

Reverse-mode computes *all* gradients in one backward pass, regardless of how many inputs exist. This is why it's preferred for neural networks (many parameters, scalar loss).

### Jacobian and the Chain Rule

For vector-valued functions, the chain rule generalizes to matrix multiplication.

If f: ℝⁿ → ℝᵐ and g: ℝᵐ → ℝᵖ, with h = g ∘ f, then:

```
Jₕ(x) = Jg(f(x)) · Jf(x)
```

Where Jf is the m×n Jacobian matrix:

```
        ⎡ ∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ ⎤
Jf(x) = ⎢ ∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ ⎥
        ⎣   ...       ...    ...    ...   ⎦
```

In reverse-mode autodiff, we compute **vector-Jacobian products** (VJPs):

```
v̄ · Jf    (row vector times Jacobian)
```

This is efficient because we never form the full Jacobian — we compute the product directly.

### Connection to Paladium

In Paladium's `Grad.backward`:

```scala
case Mul(l, r) =>
  val a1 = backwardAccum(l, num.times(upstream, r.eval), accum)
  backwardAccum(r, num.times(upstream, l.eval), a1)
```

This implements:

```
∂f         ∂f      ∂(l·r)
───── = ─────── · ─────── = upstream · r    (for left child)
∂l      ∂(l·r)      ∂l

∂f         ∂f      ∂(l·r)
───── = ─────── · ─────── = upstream · l    (for right child)
∂r      ∂(l·r)      ∂r
```

The `upstream` variable carries the adjoint (v̄) through the computation graph, accumulating gradients via the chain rule at each node.

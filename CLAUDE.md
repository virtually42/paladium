# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

Mill build system with Scala 3.6.3:

```bash
mill paladium.compile              # Compile
mill paladium.test                 # Run all tests
mill paladium.test.testOnly paladium.ValueSuite  # Run single suite
```

## Architecture

Interpreter pattern for reverse-mode automatic differentiation. The `Value[A]` ADT represents computations; multiple interpreters walk the tree for different purposes.

```
┌─────────────────────────────────────────────────────────┐
│  Interpreters                                           │
│  • eval           → A                  (compute value)  │
│  • Grad.backward  → Map[String, A]     (numerical grad) │
│  • SymbolicGrad   → Map[String, Value[A]] (grad exprs)  │
├─────────────────────────────────────────────────────────┤
│  AutoGrad   (convenience layer)                         │
│  expr.trace → Traced(value, grads)                      │
├─────────────────────────────────────────────────────────┤
│  Value[A]   (expression tree ADT)                       │
│  Var | Lit | Const | Add | Sub | Mul | Div | Pow | Neg | Log │
└─────────────────────────────────────────────────────────┘
```

**Value** (`Value.scala`): Expression tree enum. Operations build tree nodes; `eval` walks the tree.
- `Const(n: Int)` - structural constants (coefficients in formulas)
- `Lit(data: A)` - concrete typed values
- `Var(id, data)` - gradient-tracked variables

**Grad** (`Grad.scala`): Reverse-mode autodiff computing numerical gradients. Walks tree backward applying chain rule with accumulator pattern.

**SymbolicGrad** (`SymbolicGrad.scala`): Builds gradient *expressions* instead of computing values. Returns `Map[String, Value[A]]` for deferred execution, GPU compilation, or visualization.

**AutoGrad** (`AutoGrad.scala`): Bundles `eval` + `backward` into `Traced[A](value, grads)`.

**NumberLike** (`NumberLike.scala`): Typeclass for numeric operations. Implement to support new numeric types.

**Dsl** (`Dsl.scala`): Ergonomic syntax extensions.
- `val x = 2.0.^` creates `Var("x", 2.0)` with name captured from binding
- Implicit `Conversion[A, Value[A]]` allows `x + 2.0` without wrapping

## Conventions

- Extension methods live in nested `object syntax` (e.g., `import Grad.syntax.*`)
- Tests verify gradients against finite differences
- `Const` for structural integers in formulas; `Lit` for concrete typed data

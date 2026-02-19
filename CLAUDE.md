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

Three-layer design for reverse-mode automatic differentiation:

```
┌─────────────────────────────────────────┐
│  AutoGrad   (convenience layer)         │  expr.trace → Traced(value, grads)
├─────────────────────────────────────────┤
│  Grad       (differentiation)           │  Grad.backward(expr) → Map[String, A]
├─────────────────────────────────────────┤
│  Value      (expression tree)           │  Build & evaluate expressions
└─────────────────────────────────────────┘
```

**Value** (`paladium/src/paladium/Value.scala`): Expression tree enum. Operations build tree nodes; `eval` walks the tree. Three value types:
- `Const(n: Int)` - structural constants (coefficients in formulas)
- `Lit(data: A)` - concrete typed values
- `Var(id, data)` - gradient-tracked variables

**Grad** (`paladium/src/paladium/Grad.scala`): Reverse-mode autodiff via accumulator pattern. Walks tree backward applying chain rule.

**AutoGrad** (`paladium/src/paladium/AutoGrad.scala`): Bundles `eval` + `backward` into `Traced[A](value, grads)`.

**NumberLike** (`paladium/src/paladium/NumberLike.scala`): Type-class for numeric operations. Implement this trait to support new numeric types.

## Conventions

- Extension methods live in nested `object syntax` (e.g., `import Grad.syntax.*`)
- Tests use numerical gradient verification against finite differences

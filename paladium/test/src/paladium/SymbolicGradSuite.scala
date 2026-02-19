package paladium

import munit.FunSuite
import Value.*

class SymbolicGradSuite extends FunSuite:
  import SymbolicGrad.syntax.*

  test("symbolic gradient of x returns upstream"):
    val x = Var("x", 3.0)
    val grads = x.symbolicBackward
    assertEquals(grads("x").eval, 1.0)

  test("symbolic gradient of x + y"):
    val x = Var("x", 2.0)
    val y = Var("y", 3.0)
    val expr = x + y
    val grads = expr.symbolicBackward
    assertEquals(grads("x").eval, 1.0)
    assertEquals(grads("y").eval, 1.0)

  test("symbolic gradient of x * y"):
    val x = Var("x", 2.0)
    val y = Var("y", 3.0)
    val expr = x * y
    val grads = expr.symbolicBackward
    // d/dx (x*y) = y = 3.0
    // d/dy (x*y) = x = 2.0
    assertEquals(grads("x").eval, 3.0)
    assertEquals(grads("y").eval, 2.0)

  test("symbolic gradient of x^2"):
    val x = Var("x", 3.0)
    val expr = x ** Const(2)
    val grads = expr.symbolicBackward
    // d/dx (x^2) = 2x = 6.0
    assertEquals(grads("x").eval, 6.0)

  test("symbolic gradient matches numerical gradient for polynomial"):
    val x = Var("x", 2.0)
    // f(x) = 3x^2 - 4x + 5
    // f'(x) = 6x - 4 = 8 at x=2
    val expr = Const[Double](3) * (x ** Const(2)) - Const[Double](4) * x + Const(5)

    val symbolicGrads = expr.symbolicBackward
    val numericalGrads = Grad.backward(expr)

    assertEqualsDouble(symbolicGrads("x").eval, numericalGrads("x"), 1e-10)

  test("symbolic gradient of x / y"):
    val x = Var("x", 6.0)
    val y = Var("y", 2.0)
    val expr = x / y
    val grads = expr.symbolicBackward
    // d/dx (x/y) = 1/y = 0.5
    // d/dy (x/y) = -x/y^2 = -6/4 = -1.5
    assertEqualsDouble(grads("x").eval, 0.5, 1e-10)
    assertEqualsDouble(grads("y").eval, -1.5, 1e-10)

  test("symbolic gradient of -x"):
    val x = Var("x", 5.0)
    val expr = -x
    val grads = expr.symbolicBackward
    assertEquals(grads("x").eval, -1.0)

  test("symbolic gradient of log(x)"):
    val x = Var("x", 2.0)
    val expr = x.log
    val grads = expr.symbolicBackward
    // d/dx log(x) = 1/x = 0.5
    assertEquals(grads("x").eval, 0.5)

  test("symbolic gradient preserves expression structure"):
    val x = Var("x", 2.0)
    val y = Var("y", 3.0)
    val expr = x * y
    val grads = expr.symbolicBackward
    // The gradient for x should be Mul(upstream, y) = Mul(Const(1), Var("y", 3.0))
    // After Add with initial Const(0), it's Add(Const(0), Mul(Const(1), Var("y", 3.0)))
    grads("x") match
      case Add(Const(0), Mul(Const(1), Var("y", _))) => ()
      case other => fail(s"Unexpected expression structure: $other")

  test("symbolic and numerical gradients match for complex expression"):
    val x = Var("x", 2.0)
    val y = Var("y", 3.0)
    // f(x,y) = x^2 * y + x / y
    val expr = (x ** Const(2)) * y + x / y

    val symbolicGrads = expr.symbolicBackward
    val numericalGrads = Grad.backward(expr)

    assertEqualsDouble(symbolicGrads("x").eval, numericalGrads("x"), 1e-10)
    assertEqualsDouble(symbolicGrads("y").eval, numericalGrads("y"), 1e-10)

  test("symbolic gradient with variable exponent"):
    val x = Var("x", 2.0)
    val y = Var("y", 3.0)
    // f(x,y) = x^y
    val expr = x ** y

    val symbolicGrads = expr.symbolicBackward
    val numericalGrads = Grad.backward(expr)

    // d/dx (x^y) = y * x^(y-1) = 3 * 2^2 = 12
    // d/dy (x^y) = x^y * log(x) = 8 * log(2)
    assertEqualsDouble(symbolicGrads("x").eval, numericalGrads("x"), 1e-10)
    assertEqualsDouble(symbolicGrads("y").eval, numericalGrads("y"), 1e-10)

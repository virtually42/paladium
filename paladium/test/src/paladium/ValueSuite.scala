package paladium

class ValueSuite extends munit.FunSuite:

  test("Lit evaluates to its value") {
    val x = Value(42.0)
    assertEquals(x.eval, 42.0)
  }

  test("Const evaluates via fromInt") {
    val x = Value.const[Double](42)
    assertEquals(x.eval, 42.0)
  }

  test("Add evaluates correctly") {
    val x = Value(2.0)
    val y = Value(3.0)
    assertEquals((x + y).eval, 5.0)
  }

  test("Sub evaluates correctly") {
    val x = Value(5.0)
    val y = Value(3.0)
    assertEquals((x - y).eval, 2.0)
  }

  test("Mul evaluates correctly") {
    val x = Value(4.0)
    val y = Value(5.0)
    assertEquals((x * y).eval, 20.0)
  }

  test("Div evaluates correctly") {
    val x = Value(10.0)
    val y = Value(2.0)
    assertEquals((x / y).eval, 5.0)
  }

  test("Pow with ** evaluates correctly") {
    val x = Value(2.0)
    val y = Value(3.0)
    assertEquals((x ** y).eval, 8.0)
  }

  test("Pow with .pow evaluates correctly") {
    val x = Value(2.0)
    val y = Value(3.0)
    assertEquals(x.pow(y).eval, 8.0)
  }

  test("Neg evaluates correctly") {
    val x = Value(5.0)
    assertEquals((-x).eval, -5.0)
  }

  test("compound expression: (2 + 3) * 4") {
    val result = (Value(2.0) + Value(3.0)) * Value(4.0)
    assertEquals(result.eval, 20.0)
  }

  test("compound expression: 10 / (2 + 3)") {
    val result = Value(10.0) / (Value(2.0) + Value(3.0))
    assertEquals(result.eval, 2.0)
  }

  test("nested expression: ((1 + 2) * (3 + 4)) / 7") {
    val a = Value(1.0) + Value(2.0)  // 3
    val b = Value(3.0) + Value(4.0)  // 7
    val c = (a * b) / Value(7.0)     // 21 / 7 = 3
    assertEquals(c.eval, 3.0)
  }

  test("expression tree structure is preserved") {
    val x = Value(2.0)
    val y = Value(3.0)
    val expr = x + y

    expr match
      case Value.Add(Value.Lit(l), Value.Lit(r)) =>
        assertEquals(l, 2.0)
        assertEquals(r, 3.0)
      case _ =>
        fail("Expected Add(Lit, Lit)")
  }

  test("works with Float") {
    val x = Value(2.0f)
    val y = Value(3.0f)
    assertEquals((x + y).eval, 5.0f)
  }

  // Polynomial: f(x) = 3x^2 - 4x + 5
  def f(x: Value[Double]): Value[Double] =
    Value.const(3) * (x ** Value.const(2)) - Value.const(4) * x + Value.const(5)

  test("polynomial f(x) = 3x^2 - 4x + 5 at x=0") {
    // f(0) = 0 - 0 + 5 = 5
    assertEquals(f(Value(0.0)).eval, 5.0)
  }

  test("polynomial f(x) = 3x^2 - 4x + 5 at x=1") {
    // f(1) = 3 - 4 + 5 = 4
    assertEquals(f(Value(1.0)).eval, 4.0)
  }

  test("polynomial f(x) = 3x^2 - 4x + 5 at x=2") {
    // f(2) = 12 - 8 + 5 = 9
    assertEquals(f(Value(2.0)).eval, 9.0)
  }

  test("polynomial f(x) = 3x^2 - 4x + 5 at x=-1") {
    // f(-1) = 3 + 4 + 5 = 12
    assertEquals(f(Value(-1.0)).eval, 12.0)
  }

  test("polynomial expression tree uses Const") {
    val expr = f(Value(2.0))

    // The tree should contain Const nodes
    def containsConst(v: Value[Double]): Boolean = v match
      case Value.Const(_)      => true
      case Value.Add(l, r)     => containsConst(l) || containsConst(r)
      case Value.Sub(l, r)     => containsConst(l) || containsConst(r)
      case Value.Mul(l, r)     => containsConst(l) || containsConst(r)
      case Value.Div(l, r)     => containsConst(l) || containsConst(r)
      case Value.Pow(b, e)     => containsConst(b) || containsConst(e)
      case Value.Neg(v)        => containsConst(v)
      case Value.Lit(_)        => false

    assert(containsConst(expr), "Expression should contain Const nodes")
  }

package paladium

enum Value[A]:
  case Var(id: String, data: A)
  case Lit(data: A)
  case Const(n: Int)
  case Add(left: Value[A], right: Value[A])
  case Sub(left: Value[A], right: Value[A])
  case Mul(left: Value[A], right: Value[A])
  case Div(left: Value[A], right: Value[A])
  case Pow(base: Value[A], exp: Value[A])
  case Neg(value: Value[A])

  def +(other: Value[A]): Value[A] = Add(this, other)
  def -(other: Value[A]): Value[A] = Sub(this, other)
  def *(other: Value[A]): Value[A] = Mul(this, other)
  def /(other: Value[A]): Value[A] = Div(this, other)
  def **(exp: Value[A]): Value[A] = Pow(this, exp)
  def pow(exp: Value[A]): Value[A] = Pow(this, exp)
  def unary_- : Value[A] = Neg(this)

  def eval(using num: NumberLike[A]): A =
    this match
      case Var(_, data)     => data
      case Lit(data)        => data
      case Const(n)         => num.fromInt(n)
      case Add(left, right) => num.plus(left.eval, right.eval)
      case Sub(left, right) => num.minus(left.eval, right.eval)
      case Mul(left, right) => num.times(left.eval, right.eval)
      case Div(left, right) => num.div(left.eval, right.eval)
      case Pow(base, exp)   => num.pow(base.eval, exp.eval)
      case Neg(value)       => num.minus(num.fromInt(0), value.eval)

object Value:
  def apply[A](data: A): Value[A] = Lit(data)
  def const[A](n: Int): Value[A] = Const(n)
  def variable[A](id: String, data: A): Value[A] = Var(id, data)

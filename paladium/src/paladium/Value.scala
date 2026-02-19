package paladium

enum Value[A]:
  case Lit(data: A)
  case Add(left: Value[A], right: Value[A])
  case Mul(left: Value[A], right: Value[A])
  case Div(left: Value[A], right: Value[A])

  def +(other: Value[A]): Value[A] = Add(this, other)
  def *(other: Value[A]): Value[A] = Mul(this, other)
  def /(other: Value[A]): Value[A] = Div(this, other)

  def eval(using frac: Fractional[A]): A =
    import frac.*
    this match
      case Lit(data)        => data
      case Add(left, right) => left.eval + right.eval
      case Mul(left, right) => left.eval * right.eval
      case Div(left, right) => left.eval / right.eval

object Value:
  def apply[A](data: A): Value[A] = Lit(data)

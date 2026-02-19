package paladium

trait NumberLike[A]:
  def plus(x: A, y: A): A
  def minus(x: A, y: A): A
  def times(x: A, y: A): A
  def div(x: A, y: A): A
  def pow(x: A, exp: A): A
  def fromInt(n: Int): A

object NumberLike:
  given NumberLike[Double] with
    def plus(x: Double, y: Double): Double = x + y
    def minus(x: Double, y: Double): Double = x - y
    def times(x: Double, y: Double): Double = x * y
    def div(x: Double, y: Double): Double = x / y
    def pow(x: Double, exp: Double): Double = math.pow(x, exp)
    def fromInt(n: Int): Double = n.toDouble

  given NumberLike[Float] with
    def plus(x: Float, y: Float): Float = x + y
    def minus(x: Float, y: Float): Float = x - y
    def times(x: Float, y: Float): Float = x * y
    def div(x: Float, y: Float): Float = x / y
    def pow(x: Float, exp: Float): Float = math.pow(x.toDouble, exp.toDouble).toFloat
    def fromInt(n: Int): Float = n.toFloat

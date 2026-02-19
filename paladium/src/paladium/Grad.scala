package paladium

object Grad:
  /** Compute gradients for all variables in the expression via reverse-mode autodiff */
  def backward[A: NumberLike](expr: Value[A]): Map[String, A] =
    val num = summon[NumberLike[A]]
    backwardAccum(expr, num.fromInt(1), Map.empty)

  private def backwardAccum[A](
      expr: Value[A],
      upstream: A,
      accum: Map[String, A]
  )(using num: NumberLike[A]): Map[String, A] =
    import Value.*
    expr match
      case Var(id, _) =>
        val existing = accum.getOrElse(id, num.fromInt(0))
        accum.updated(id, num.plus(existing, upstream))

      case Lit(_) | Const(_) =>
        accum // constants have no gradient

      case Add(l, r) =>
        // d/dl (l + r) = 1, d/dr (l + r) = 1
        val a1 = backwardAccum(l, upstream, accum)
        backwardAccum(r, upstream, a1)

      case Sub(l, r) =>
        // d/dl (l - r) = 1, d/dr (l - r) = -1
        val a1 = backwardAccum(l, upstream, accum)
        val negUpstream = num.minus(num.fromInt(0), upstream)
        backwardAccum(r, negUpstream, a1)

      case Mul(l, r) =>
        // d/dl (l * r) = r, d/dr (l * r) = l
        val a1 = backwardAccum(l, num.times(upstream, r.eval), accum)
        backwardAccum(r, num.times(upstream, l.eval), a1)

      case Div(l, r) =>
        // d/dl (l / r) = 1/r
        // d/dr (l / r) = -l / r^2
        val rVal = r.eval
        val lVal = l.eval
        val a1 = backwardAccum(l, num.div(upstream, rVal), accum)
        val rGrad = num.minus(
          num.fromInt(0),
          num.div(num.times(upstream, lVal), num.times(rVal, rVal))
        )
        backwardAccum(r, rGrad, a1)

      case Pow(base, exp) =>
        // d/dbase (base^exp) = exp * base^(exp-1)
        // d/dexp (base^exp) = base^exp * log(base)
        val baseVal = base.eval
        val expVal = exp.eval
        val powVal = num.pow(baseVal, expVal)

        val baseGrad = num.times(
          upstream,
          num.times(expVal, num.pow(baseVal, num.minus(expVal, num.fromInt(1))))
        )
        val a1 = backwardAccum(base, baseGrad, accum)

        val expGrad = num.times(upstream, num.times(powVal, num.log(baseVal)))
        backwardAccum(exp, expGrad, a1)

      case Neg(v) =>
        // d/dv (-v) = -1
        val negUpstream = num.minus(num.fromInt(0), upstream)
        backwardAccum(v, negUpstream, accum)

  object syntax:
    extension [A: NumberLike](expr: Value[A])
      def backward: Map[String, A] = Grad.backward(expr)

package paladium

object SymbolicGrad:
  /** Build symbolic gradient expressions for all variables via reverse-mode autodiff */
  def backward[A](expr: Value[A]): Map[String, Value[A]] =
    backwardAccum(expr, Value.const(1), Map.empty)

  private def backwardAccum[A](
      expr: Value[A],
      upstream: Value[A],
      accum: Map[String, Value[A]]
  ): Map[String, Value[A]] =
    import Value.*
    expr match
      case Var(id, _) =>
        val existing = accum.getOrElse(id, Const(0))
        accum.updated(id, Add(existing, upstream))

      case Lit(_) | Const(_) =>
        accum // constants have no gradient

      case Add(l, r) =>
        // d/dl (l + r) = 1, d/dr (l + r) = 1
        val a1 = backwardAccum(l, upstream, accum)
        backwardAccum(r, upstream, a1)

      case Sub(l, r) =>
        // d/dl (l - r) = 1, d/dr (l - r) = -1
        val a1 = backwardAccum(l, upstream, accum)
        backwardAccum(r, Neg(upstream), a1)

      case Mul(l, r) =>
        // d/dl (l * r) = r, d/dr (l * r) = l
        val a1 = backwardAccum(l, Mul(upstream, r), accum)
        backwardAccum(r, Mul(upstream, l), a1)

      case Div(l, r) =>
        // d/dl (l / r) = 1/r
        // d/dr (l / r) = -l / r^2
        val a1 = backwardAccum(l, Div(upstream, r), accum)
        val rGrad = Neg(Div(Mul(upstream, l), Mul(r, r)))
        backwardAccum(r, rGrad, a1)

      case Pow(base, exp) =>
        // d/dbase (base^exp) = exp * base^(exp-1)
        // d/dexp (base^exp) = base^exp * log(base)
        val baseGrad = Mul(upstream, Mul(exp, Pow(base, Sub(exp, Const(1)))))
        val a1 = backwardAccum(base, baseGrad, accum)

        val expGrad = Mul(upstream, Mul(Pow(base, exp), Log(base)))
        backwardAccum(exp, expGrad, a1)

      case Neg(v) =>
        // d/dv (-v) = -1
        backwardAccum(v, Neg(upstream), accum)

      case Log(v) =>
        // d/dv log(v) = 1/v
        backwardAccum(v, Div(upstream, v), accum)

  object syntax:
    extension [A](expr: Value[A])
      def symbolicBackward: Map[String, Value[A]] = SymbolicGrad.backward(expr)

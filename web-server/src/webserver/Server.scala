package webserver

import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.server.netty.sync.NettySyncServer
import sttp.tapir.json.circe.*
import io.circe.generic.auto.*
import paladium.*

// Request/Response models
case class EvaluateRequest(expression: String, variables: Map[String, Double])
case class EvaluateResponse(result: Double)

case class GradientRequest(expression: String, variables: Map[String, Double])
case class GradientResponse(value: Double, gradients: Map[String, Double])

case class SymbolicGradientRequest(expression: String)
case class SymbolicGradientResponse(gradients: Map[String, String])

case class HealthResponse(status: String, version: String)

object Server:

  // Health check endpoint
  val healthEndpoint: PublicEndpoint[Unit, Unit, HealthResponse, Any] =
    endpoint.get
      .in("api" / "health")
      .out(jsonBody[HealthResponse])

  // Evaluate expression endpoint
  val evaluateEndpoint: PublicEndpoint[EvaluateRequest, String, EvaluateResponse, Any] =
    endpoint.post
      .in("api" / "evaluate")
      .in(jsonBody[EvaluateRequest])
      .out(jsonBody[EvaluateResponse])
      .errorOut(stringBody)

  // Compute numerical gradients endpoint
  val gradientEndpoint: PublicEndpoint[GradientRequest, String, GradientResponse, Any] =
    endpoint.post
      .in("api" / "gradient")
      .in(jsonBody[GradientRequest])
      .out(jsonBody[GradientResponse])
      .errorOut(stringBody)

  // Compute symbolic gradients endpoint
  val symbolicGradientEndpoint: PublicEndpoint[SymbolicGradientRequest, String, SymbolicGradientResponse, Any] =
    endpoint.post
      .in("api" / "symbolic-gradient")
      .in(jsonBody[SymbolicGradientRequest])
      .out(jsonBody[SymbolicGradientResponse])
      .errorOut(stringBody)

  // Health check logic
  def healthLogic(unit: Unit): HealthResponse =
    HealthResponse(status = "ok", version = "0.1.0")

  // Expression parser (simple DSL for demo)
  // Format: "x * y + z" where variables are single letters
  def parseExpression(expr: String, vars: Map[String, Double]): Either[String, Value[Double]] =
    try
      // Simple recursive descent parser for demo
      val tokens = tokenize(expr)
      val (result, _) = parseAddSub(tokens, vars)
      Right(result)
    catch
      case e: Exception => Left(s"Parse error: ${e.getMessage}")

  private def tokenize(expr: String): List[String] =
    expr
      .replaceAll("\\s+", "")
      .replaceAll("([+\\-*/()^])", " $1 ")
      .split("\\s+")
      .filter(_.nonEmpty)
      .toList

  private def parseAddSub(tokens: List[String], vars: Map[String, Double]): (Value[Double], List[String]) =
    var (left, remaining) = parseMulDiv(tokens, vars)
    var current = remaining
    while current.nonEmpty && (current.head == "+" || current.head == "-") do
      val op = current.head
      val (right, rest) = parseMulDiv(current.tail, vars)
      left = if op == "+" then left + right else left - right
      current = rest
    (left, current)

  private def parseMulDiv(tokens: List[String], vars: Map[String, Double]): (Value[Double], List[String]) =
    var (left, remaining) = parsePow(tokens, vars)
    var current = remaining
    while current.nonEmpty && (current.head == "*" || current.head == "/") do
      val op = current.head
      val (right, rest) = parsePow(current.tail, vars)
      left = if op == "*" then left * right else left / right
      current = rest
    (left, current)

  private def parsePow(tokens: List[String], vars: Map[String, Double]): (Value[Double], List[String]) =
    val (base, remaining) = parseUnary(tokens, vars)
    if remaining.nonEmpty && remaining.head == "^" then
      val (exp, rest) = parsePow(remaining.tail, vars) // right-associative
      (base ** exp, rest)
    else
      (base, remaining)

  private def parseUnary(tokens: List[String], vars: Map[String, Double]): (Value[Double], List[String]) =
    tokens match
      case "-" :: rest =>
        val (value, remaining) = parseUnary(rest, vars)
        (-value, remaining)
      case _ =>
        parsePrimary(tokens, vars)

  private def parsePrimary(tokens: List[String], vars: Map[String, Double]): (Value[Double], List[String]) =
    tokens match
      case "(" :: rest =>
        val (expr, afterExpr) = parseAddSub(rest, vars)
        afterExpr match
          case ")" :: remaining => (expr, remaining)
          case _ => throw Exception("Missing closing parenthesis")
      case "log" :: "(" :: rest =>
        val (expr, afterExpr) = parseAddSub(rest, vars)
        afterExpr match
          case ")" :: remaining => (expr.log, remaining)
          case _ => throw Exception("Missing closing parenthesis for log")
      case token :: rest =>
        val value = token.toDoubleOption match
          case Some(d) => Value.Lit(d)
          case None =>
            vars.get(token) match
              case Some(v) => Value.Var(token, v)
              case None => throw Exception(s"Unknown variable: $token")
        (value, rest)
      case Nil =>
        throw Exception("Unexpected end of expression")

  // Pretty-print Value expression as string
  def valueToString[A](v: Value[A]): String =
    import Value.*
    v match
      case Var(id, _)     => id
      case Lit(data)      => data.toString
      case Const(n)       => n.toString
      case Add(l, r)      => s"(${valueToString(l)} + ${valueToString(r)})"
      case Sub(l, r)      => s"(${valueToString(l)} - ${valueToString(r)})"
      case Mul(l, r)      => s"(${valueToString(l)} * ${valueToString(r)})"
      case Div(l, r)      => s"(${valueToString(l)} / ${valueToString(r)})"
      case Pow(base, exp) => s"(${valueToString(base)} ^ ${valueToString(exp)})"
      case Neg(value)     => s"(-${valueToString(value)})"
      case Log(value)     => s"log(${valueToString(value)})"

  // Evaluate expression logic
  def evaluateLogic(req: EvaluateRequest): Either[String, EvaluateResponse] =
    parseExpression(req.expression, req.variables).map { expr =>
      EvaluateResponse(expr.eval)
    }

  // Gradient computation logic
  def gradientLogic(req: GradientRequest): Either[String, GradientResponse] =
    parseExpression(req.expression, req.variables).map { expr =>
      val value = expr.eval
      val grads = Grad.backward(expr)
      GradientResponse(value, grads)
    }

  // Symbolic gradient computation logic
  def symbolicGradientLogic(req: SymbolicGradientRequest): Either[String, SymbolicGradientResponse] =
    // For symbolic gradients, we don't need actual values, just variable names
    // Extract variable names from expression and create placeholder vars
    val varPattern = "[a-zA-Z_][a-zA-Z0-9_]*".r
    val varNames = varPattern.findAllIn(req.expression).toSet - "log" // exclude function names
    val vars = varNames.map(name => name -> 0.0).toMap

    parseExpression(req.expression, vars).map { expr =>
      val symbolicGrads = SymbolicGrad.backward(expr)
      val gradStrings = symbolicGrads.map { case (k, v) => k -> valueToString(v) }
      SymbolicGradientResponse(gradStrings)
    }

  def main(args: Array[String]): Unit =
    val port = sys.env.getOrElse("PORT", "8080").toInt

    val server = NettySyncServer()
      .port(port)
      .addEndpoint(healthEndpoint.handleSuccess(healthLogic))
      .addEndpoint(evaluateEndpoint.handle(evaluateLogic))
      .addEndpoint(gradientEndpoint.handle(gradientLogic))
      .addEndpoint(symbolicGradientEndpoint.handle(symbolicGradientLogic))

    println(s"Starting server on port $port...")
    println("Endpoints:")
    println("  GET  /api/health")
    println("  POST /api/evaluate")
    println("  POST /api/gradient")
    println("  POST /api/symbolic-gradient")

    server.startAndWait()

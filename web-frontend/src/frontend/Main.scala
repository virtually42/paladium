package frontend

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import org.scalajs.dom
import paladium.*

/** Scala.js exports for Vite integration.
  *
  * These functions are exported to JavaScript and can be called from the Vite
  * frontend to interact with the paladium library directly in the browser.
  */
@JSExportTopLevel("PaladiumFrontend")
object Main:

  // Import the given NumberLike[Double] from the companion object

  /** Evaluate a simple expression with the given variable values.
    *
    * @param expr
    *   Expression string (e.g., "x * y + z")
    * @param variables
    *   JavaScript object mapping variable names to values
    * @return
    *   The computed result
    */
  @JSExport
  def evaluateExpression(expr: String, variables: js.Dictionary[Double]): Double =
    val vars = variables.toMap
    val parsed = parseExpression(expr, vars)
    parsed.eval

  /** Get the symbolic gradient expressions for all variables in an expression.
    *
    * @param expr
    *   Expression string (e.g., "x^2 + y^2")
    * @return
    *   JavaScript object mapping variable names to gradient expression strings
    */
  @JSExport
  def getSymbolicGradients(expr: String): js.Dictionary[String] =
    val varPattern = "[a-zA-Z_][a-zA-Z0-9_]*".r
    val varNames = varPattern.findAllIn(expr).toSet - "log"
    val vars = varNames.map(name => name -> 0.0).toMap

    val parsed = parseExpression(expr, vars)
    val grads = SymbolicGrad.backward(parsed)

    js.Dictionary(grads.map { case (k, v) => k -> valueToString(v) }.toSeq*)

  /** Get numerical gradients for all variables.
    *
    * @param expr
    *   Expression string
    * @param variables
    *   JavaScript object mapping variable names to values
    * @return
    *   JavaScript object with 'value' and 'gradients' properties
    */
  @JSExport
  def getGradients(expr: String, variables: js.Dictionary[Double]): js.Dynamic =
    val vars = variables.toMap
    val parsed = parseExpression(expr, vars)
    val value = parsed.eval
    val grads = Grad.backward(parsed)

    js.Dynamic.literal(
      value = value,
      gradients = js.Dictionary(grads.toSeq*)
    )

  /** Convert an expression to a Mermaid graph representation for visualization.
    *
    * @param expr
    *   Expression string
    * @return
    *   Mermaid graph definition string
    */
  @JSExport
  def toMermaidGraph(expr: String): String =
    val varPattern = "[a-zA-Z_][a-zA-Z0-9_]*".r
    val varNames = varPattern.findAllIn(expr).toSet - "log"
    val vars = varNames.map(name => name -> 0.0).toMap

    val parsed = parseExpression(expr, vars)
    generateMermaid(parsed)

  /** Convert an expression to a D3-compatible JSON graph representation.
    *
    * @param expr
    *   Expression string
    * @return
    *   JSON string with nodes and links arrays
    */
  @JSExport
  def toD3Graph(expr: String): String =
    val varPattern = "[a-zA-Z_][a-zA-Z0-9_]*".r
    val varNames = varPattern.findAllIn(expr).toSet - "log"
    val vars = varNames.map(name => name -> 0.0).toMap

    val parsed = parseExpression(expr, vars)
    generateD3Json(parsed)

  // Simple expression parser
  private def parseExpression(expr: String, vars: Map[String, Double]): Value[Double] =
    val tokens = tokenize(expr)
    val (result, _) = parseAddSub(tokens, vars)
    result

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
      val (exp, rest) = parsePow(remaining.tail, vars)
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
          case _                => throw Exception("Missing closing parenthesis")
      case "log" :: "(" :: rest =>
        val (expr, afterExpr) = parseAddSub(rest, vars)
        afterExpr match
          case ")" :: remaining => (expr.log, remaining)
          case _                => throw Exception("Missing closing parenthesis for log")
      case token :: rest =>
        val value = token.toDoubleOption match
          case Some(d) => Value.Lit(d)
          case None =>
            vars.get(token) match
              case Some(v) => Value.Var(token, v)
              case None    => throw Exception(s"Unknown variable: $token")
        (value, rest)
      case Nil =>
        throw Exception("Unexpected end of expression")

  private def valueToString[A](v: Value[A]): String =
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

  // Generate Mermaid graph from expression tree
  private def generateMermaid[A](expr: Value[A]): String =
    val sb = new StringBuilder("graph TD\n")
    var nodeId = 0

    def nextId(): String =
      val id = s"n$nodeId"
      nodeId += 1
      id

    def visit(v: Value[A]): String =
      import Value.*
      val id = nextId()
      v match
        case Var(name, _) =>
          sb.append(s"    $id[$name]\n")
          id
        case Lit(data) =>
          sb.append(s"    $id[$data]\n")
          id
        case Const(n) =>
          sb.append(s"    $id[$n]\n")
          id
        case Add(l, r) =>
          val lId = visit(l)
          val rId = visit(r)
          sb.append(s"    $id((+))\n")
          sb.append(s"    $lId --> $id\n")
          sb.append(s"    $rId --> $id\n")
          id
        case Sub(l, r) =>
          val lId = visit(l)
          val rId = visit(r)
          sb.append(s"    $id((--))\n")
          sb.append(s"    $lId --> $id\n")
          sb.append(s"    $rId --> $id\n")
          id
        case Mul(l, r) =>
          val lId = visit(l)
          val rId = visit(r)
          sb.append(s"    $id((**))\n")
          sb.append(s"    $lId --> $id\n")
          sb.append(s"    $rId --> $id\n")
          id
        case Div(l, r) =>
          val lId = visit(l)
          val rId = visit(r)
          sb.append(s"    $id((//))\n")
          sb.append(s"    $lId --> $id\n")
          sb.append(s"    $rId --> $id\n")
          id
        case Pow(base, exp) =>
          val bId = visit(base)
          val eId = visit(exp)
          sb.append(s"    $id((^))\n")
          sb.append(s"    $bId --> $id\n")
          sb.append(s"    $eId --> $id\n")
          id
        case Neg(v) =>
          val vId = visit(v)
          sb.append(s"    $id((--))\n")
          sb.append(s"    $vId --> $id\n")
          id
        case Log(v) =>
          val vId = visit(v)
          sb.append(s"    $id((log))\n")
          sb.append(s"    $vId --> $id\n")
          id

    visit(expr)
    sb.toString

  // Generate D3-compatible JSON graph
  private def generateD3Json[A](expr: Value[A]): String =
    import scala.collection.mutable.ArrayBuffer

    case class Node(id: Int, label: String, nodeType: String)
    case class Link(source: Int, target: Int)

    val nodes = ArrayBuffer[Node]()
    val links = ArrayBuffer[Link]()
    var nodeId = 0

    def nextId(): Int =
      val id = nodeId
      nodeId += 1
      id

    def visit(v: Value[A]): Int =
      import Value.*
      val id = nextId()
      v match
        case Var(name, _) =>
          nodes += Node(id, name, "variable")
          id
        case Lit(data) =>
          nodes += Node(id, data.toString, "literal")
          id
        case Const(n) =>
          nodes += Node(id, n.toString, "constant")
          id
        case Add(l, r) =>
          nodes += Node(id, "+", "operation")
          val lId = visit(l)
          val rId = visit(r)
          links += Link(lId, id)
          links += Link(rId, id)
          id
        case Sub(l, r) =>
          nodes += Node(id, "-", "operation")
          val lId = visit(l)
          val rId = visit(r)
          links += Link(lId, id)
          links += Link(rId, id)
          id
        case Mul(l, r) =>
          nodes += Node(id, "*", "operation")
          val lId = visit(l)
          val rId = visit(r)
          links += Link(lId, id)
          links += Link(rId, id)
          id
        case Div(l, r) =>
          nodes += Node(id, "/", "operation")
          val lId = visit(l)
          val rId = visit(r)
          links += Link(lId, id)
          links += Link(rId, id)
          id
        case Pow(base, exp) =>
          nodes += Node(id, "^", "operation")
          val bId = visit(base)
          val eId = visit(exp)
          links += Link(bId, id)
          links += Link(eId, id)
          id
        case Neg(v) =>
          nodes += Node(id, "-", "unary")
          val vId = visit(v)
          links += Link(vId, id)
          id
        case Log(v) =>
          nodes += Node(id, "log", "function")
          val vId = visit(v)
          links += Link(vId, id)
          id

    visit(expr)

    // Build JSON string manually to avoid dependency
    val nodesJson = nodes
      .map(n => s"""{"id":${n.id},"label":"${n.label}","type":"${n.nodeType}"}""")
      .mkString("[", ",", "]")
    val linksJson = links
      .map(l => s"""{"source":${l.source},"target":${l.target}}""")
      .mkString("[", ",", "]")

    s"""{"nodes":$nodesJson,"links":$linksJson}"""

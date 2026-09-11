import { EigenvalueDecomposition, Matrix } from 'ml-matrix'

import type { PointLike } from './image-preprocessing'

export interface EllipseCoefficients {
  A: number
  B: number
  C: number
  D: number
  E: number
  F: number
}

export interface EllipseParameters {
  coefficients: EllipseCoefficients
  center: { x: number; y: number }
  axes: { major: number; minor: number }
  rotation: number
}

export interface FitEllipseOptions {
  /** Number of RANSAC iterations to perform; set to 0 to disable. */
  ransacIterations?: number
  /** Residual tolerance (in pixels) for an edge point to be considered an inlier. */
  inlierThreshold?: number
  /** Minimum number of inliers required to accept a candidate ellipse. */
  minInliers?: number
  /**
   * Optional regularisation term that is added to the scatter matrix diagonal
   * before inversion.
   */
  regularization?: number
}

const MIN_SAMPLE_SIZE = 6

export function ellipseImplicitValue(coefficients: EllipseCoefficients, point: PointLike): number {
  const { A, B, C, D, E, F } = coefficients
  const { x, y } = point
  return A * x * x + B * x * y + C * y * y + D * x + E * y + F
}

export function fitEllipse(
  points: PointLike[],
  options: FitEllipseOptions = {}
): EllipseParameters {
  const { ransacIterations = 0, inlierThreshold = 1.5, minInliers = 25 } = options

  if (points.length < MIN_SAMPLE_SIZE) {
    throw new Error('At least six edge points are required to fit an ellipse')
  }

  if (ransacIterations <= 0) {
    return fitEllipseLeastSquares(points, options)
  }

  let best: { ellipse: EllipseParameters; inliers: PointLike[] } | undefined

  for (let i = 0; i < ransacIterations; i++) {
    const sample = randomSubset(points, MIN_SAMPLE_SIZE)

    try {
      const candidate = fitEllipseLeastSquares(sample, options)
      const residuals = points.map((p) => Math.abs(ellipseImplicitValue(candidate.coefficients, p)))
      const inliers: PointLike[] = []
      for (let j = 0; j < residuals.length; j++) {
        if (residuals[j] <= inlierThreshold) {
          inliers.push(points[j])
        }
      }

      if (inliers.length >= minInliers) {
        const refined = fitEllipseLeastSquares(inliers, options)
        if (!best || inliers.length > best.inliers.length) {
          best = { ellipse: refined, inliers }
        }
      }
    } catch {
      // Ignore degenerate samples
    }
  }

  if (best) {
    return best.ellipse
  }

  return fitEllipseLeastSquares(points, options)
}

export function fitEllipseLeastSquares(
  points: PointLike[],
  options: FitEllipseOptions = {}
): EllipseParameters {
  if (points.length < MIN_SAMPLE_SIZE) {
    throw new Error('At least six edge points are required to fit an ellipse')
  }

  const design = buildDesignMatrix(points)
  const scatter = design.transpose().mmul(design)

  const regularization = options.regularization ?? 1e-8
  for (let i = 0; i < scatter.rows; i++) {
    scatter.set(i, i, scatter.get(i, i) + regularization)
  }

  const constraint = Matrix.zeros(6, 6)
  constraint.set(0, 2, 2)
  constraint.set(1, 1, -1)
  constraint.set(2, 0, 2)

  const sinv = scatter.inverse()
  const evalMatrix = sinv.mmul(constraint)
  const eig = new EigenvalueDecomposition(evalMatrix)
  const eigenvectors = eig.eigenvectorMatrix
  const eigenvalues = eig.realEigenvalues

  let bestVector: Matrix | undefined
  for (let i = 0; i < eigenvalues.length; i++) {
    const value = eigenvalues[i]
    const vector = eigenvectors.getColumnMatrix(i)
    const coeffs = toCoefficients(vector)
    if (isFinite(value) && isEllipse(coeffs)) {
      bestVector = vector
      break
    }
  }

  if (!bestVector) {
    throw new Error('Unable to determine a valid ellipse from the provided points')
  }

  const coefficients = normaliseCoefficients(toCoefficients(bestVector))
  return coefficientsToParameters(coefficients)
}

export interface IntersectionOptions {
  /** Number of samples on the parameterised ellipse. */
  samples?: number
  /** Maximum absolute implicit value considered a hit during refinement. */
  tolerance?: number
  /** Bisection refinement iterations for each intersection. */
  refinementIterations?: number
}

export interface EllipseIntersectionResult {
  points: PointLike[]
  basePlaneDiameter: number
}

export function findEllipseIntersections(
  first: EllipseParameters,
  second: EllipseParameters,
  options: IntersectionOptions = {}
): EllipseIntersectionResult {
  const samples = options.samples ?? 720
  const tolerance = options.tolerance ?? 1e-3
  const refinementIterations = options.refinementIterations ?? 12

  const intersections: PointLike[] = []
  const step = (2 * Math.PI) / samples
  let previousPoint = sampleEllipse(first, 0)
  let previousValue = ellipseImplicitValue(second.coefficients, previousPoint)

  for (let i = 1; i <= samples; i++) {
    const angle = i * step
    const currentPoint = sampleEllipse(first, angle)
    const currentValue = ellipseImplicitValue(second.coefficients, currentPoint)

    if (Math.abs(currentValue) <= tolerance) {
      pushIfUnique(intersections, currentPoint, tolerance)
    } else if (Math.sign(previousValue) !== Math.sign(currentValue)) {
      const refined = refineIntersection(
        first,
        second,
        angle - step,
        angle,
        previousValue,
        currentValue,
        refinementIterations
      )
      if (refined) {
        pushIfUnique(intersections, refined, tolerance)
      }
    }

    previousPoint = currentPoint
    previousValue = currentValue
  }

  const basePlaneDiameter = computeBasePlaneDiameter(intersections)

  return { points: intersections, basePlaneDiameter }
}

export function computeBasePlaneDiameter(points: PointLike[]): number {
  if (points.length < 2) {
    return 0
  }

  let maxDistance = 0
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const distance = Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y)
      if (distance > maxDistance) {
        maxDistance = distance
      }
    }
  }

  return maxDistance
}

function pushIfUnique(list: PointLike[], point: PointLike, tolerance: number) {
  const minDistance = tolerance * 5
  if (!list.some((p) => Math.hypot(p.x - point.x, p.y - point.y) <= minDistance)) {
    list.push(point)
  }
}

function refineIntersection(
  first: EllipseParameters,
  second: EllipseParameters,
  start: number,
  end: number,
  startValue: number,
  endValue: number,
  iterations: number
): PointLike | undefined {
  let left = start
  let right = end
  let fLeft = startValue
  let fRight = endValue

  for (let i = 0; i < iterations; i++) {
    const mid = (left + right) / 2
    const candidate = sampleEllipse(first, mid)
    const value = ellipseImplicitValue(second.coefficients, candidate)

    if (Math.abs(value) < 1e-6) {
      return candidate
    }

    if (Math.sign(fLeft) === Math.sign(value)) {
      left = mid
      fLeft = value
    } else {
      right = mid
      fRight = value
    }
  }

  return sampleEllipse(first, (left + right) / 2)
}

function sampleEllipse(ellipse: EllipseParameters, t: number): PointLike {
  const { center, axes, rotation } = ellipse
  const cosT = Math.cos(t)
  const sinT = Math.sin(t)
  const cosR = Math.cos(rotation)
  const sinR = Math.sin(rotation)

  const xLocal = axes.major * cosT
  const yLocal = axes.minor * sinT

  return {
    x: center.x + xLocal * cosR - yLocal * sinR,
    y: center.y + xLocal * sinR + yLocal * cosR
  }
}

function buildDesignMatrix(points: PointLike[]): Matrix {
  const data = new Array(points.length * 6)
  let offset = 0

  for (const point of points) {
    const { x, y } = point
    data[offset++] = x * x
    data[offset++] = x * y
    data[offset++] = y * y
    data[offset++] = x
    data[offset++] = y
    data[offset++] = 1
  }

  return Matrix.from1DArray(points.length, 6, data)
}

function toCoefficients(vector: Matrix): EllipseCoefficients {
  return {
    A: vector.get(0, 0),
    B: vector.get(1, 0),
    C: vector.get(2, 0),
    D: vector.get(3, 0),
    E: vector.get(4, 0),
    F: vector.get(5, 0)
  }
}

function normaliseCoefficients(coeffs: EllipseCoefficients): EllipseCoefficients {
  const scale = coeffs.F !== 0 ? coeffs.F : Math.hypot(coeffs.A, coeffs.C)
  if (scale === 0) {
    return coeffs
  }
  return {
    A: coeffs.A / scale,
    B: coeffs.B / scale,
    C: coeffs.C / scale,
    D: coeffs.D / scale,
    E: coeffs.E / scale,
    F: coeffs.F / scale
  }
}

function coefficientsToParameters(coeffs: EllipseCoefficients): EllipseParameters {
  const { A, B, C, D, E, F } = coeffs

  if (B * B - 4 * A * C >= 0) {
    throw new Error('Provided coefficients do not correspond to an ellipse')
  }

  const denominator = B * B - 4 * A * C
  const x0 = (2 * C * D - B * E) / denominator
  const y0 = (2 * A * E - B * D) / denominator

  const numerator = 2 *
    (A * E * E + C * D * D + F * B * B - 2 * B * D * E - A * C * F)

  const term = Math.sqrt((A - C) * (A - C) + B * B)
  const major = Math.sqrt(Math.abs(numerator / (denominator * (C + A - term))))
  const minor = Math.sqrt(Math.abs(numerator / (denominator * (C + A + term))))

  let rotation = 0.5 * Math.atan2(B, A - C)
  let majorAxis = major
  let minorAxis = minor

  if (minorAxis > majorAxis) {
    ;[majorAxis, minorAxis] = [minorAxis, majorAxis]
    rotation += Math.PI / 2
  }

  return {
    coefficients: coeffs,
    center: { x: x0, y: y0 },
    axes: { major: majorAxis, minor: minorAxis },
    rotation
  }
}

function isEllipse(coeffs: EllipseCoefficients): boolean {
  return coeffs.B * coeffs.B - 4 * coeffs.A * coeffs.C < 0
}

function randomSubset(points: PointLike[], size: number): PointLike[] {
  if (size >= points.length) {
    return [...points]
  }

  const result: PointLike[] = []
  const used = new Set<number>()
  while (result.length < size) {
    const index = Math.floor(Math.random() * points.length)
    if (!used.has(index)) {
      used.add(index)
      result.push(points[index])
    }
  }
  return result
}

import { Image } from 'image-js'

export type ImageInput =
  | string
  | ArrayBuffer
  | Buffer
  | Uint8Array
  | Image

export interface PreprocessOptions {
  /** Gaussian sigma used prior to edge detection. */
  gaussianSigma?: number
  /** Low threshold for the Canny detector (0-255). */
  cannyLowThreshold?: number
  /** High threshold for the Canny detector (0-255). */
  cannyHighThreshold?: number
  /** Optional manual threshold used to create a binary mask (0-255). */
  threshold?: number
  /** Sample every Nth edge pixel to reduce the least-squares problem size. */
  edgePointStride?: number
}

export interface PointLike {
  x: number
  y: number
}

export interface PreprocessResult {
  original: Image
  grayscale: Image
  blurred: Image
  binaryMask: Image
  edges: Image
  edgePoints: PointLike[]
}

const DEFAULT_OPTIONS: Required<Omit<PreprocessOptions, 'threshold'>> = {
  gaussianSigma: 1.6,
  cannyLowThreshold: 25,
  cannyHighThreshold: 80,
  edgePointStride: 1
}

export async function loadImage(input: ImageInput): Promise<Image> {
  if (input instanceof Image) {
    return input
  }

  if (typeof input === 'string' || input instanceof Uint8Array || input instanceof ArrayBuffer) {
    return Image.load(input as any)
  }

  if (typeof Buffer !== 'undefined' && input instanceof Buffer) {
    return Image.load(input)
  }

  throw new TypeError('Unsupported image input type')
}

export async function preprocessImage(
  input: ImageInput,
  options: PreprocessOptions = {}
): Promise<PreprocessResult> {
  const resolvedOptions = { ...DEFAULT_OPTIONS, ...options }
  const original = await loadImage(input)

  const grayscale = original.grey({ mergeAlpha: true })
  const blurred = grayscale.gaussianFilter({ sigma: resolvedOptions.gaussianSigma })
  const binaryMask = resolvedOptions.threshold
    ? blurred.threshold({ threshold: resolvedOptions.threshold })
    : blurred.threshold({ algorithm: 'otsu' })
  const edges = blurred.cannyEdgeDetector({
    lowThreshold: resolvedOptions.cannyLowThreshold,
    highThreshold: resolvedOptions.cannyHighThreshold
  })

  const edgePoints: PointLike[] = []
  const stride = Math.max(1, Math.round(resolvedOptions.edgePointStride))

  for (let y = 0; y < edges.height; y += stride) {
    for (let x = 0; x < edges.width; x += stride) {
      const pixel = edges.getPixelXY(x, y)
      const value = Array.isArray(pixel) ? pixel[0] : (pixel as number)
      if (value > 0) {
        edgePoints.push({ x, y })
      }
    }
  }

  return { original, grayscale, blurred, binaryMask, edges, edgePoints }
}

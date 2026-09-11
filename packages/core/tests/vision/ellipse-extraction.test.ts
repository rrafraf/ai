/**
 * @jest-environment node
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { findEllipseIntersections, fitEllipse } from '../../shared/vision/ellipse-fitting'
import { preprocessImage } from '../../shared/vision/image-preprocessing'

describe('vision ellipse extraction', () => {
  it('fits droplet ellipses and reports their base-plane diameter', async () => {
    const fixturePath = join(__dirname, '..', 'fixtures', 'droplet.pgm')
    const buffer = readFileSync(fixturePath)

    const preprocess = await preprocessImage(buffer, {
      edgePointStride: 2,
      gaussianSigma: 1.4,
      cannyLowThreshold: 20,
      cannyHighThreshold: 60
    })

    const midY = preprocess.original.height / 2
    const upperPoints = preprocess.edgePoints.filter((p) => p.y <= midY)
    const lowerPoints = preprocess.edgePoints.filter((p) => p.y > midY)

    expect(upperPoints.length).toBeGreaterThan(50)
    expect(lowerPoints.length).toBeGreaterThan(50)

    const upperEllipse = fitEllipse(upperPoints, {
      ransacIterations: 120,
      inlierThreshold: 2.5,
      minInliers: 60
    })
    const lowerEllipse = fitEllipse(lowerPoints, {
      ransacIterations: 120,
      inlierThreshold: 2.5,
      minInliers: 60
    })

    expect(upperEllipse.axes.major).toBeGreaterThan(40)
    expect(lowerEllipse.axes.major).toBeGreaterThan(40)

    const { points, basePlaneDiameter } = findEllipseIntersections(upperEllipse, lowerEllipse, {
      tolerance: 1e-2,
      samples: 1440,
      refinementIterations: 18
    })

    expect(points.length).toBeGreaterThanOrEqual(2)
    expect(basePlaneDiameter).toBeGreaterThan(90)
    expect(basePlaneDiameter).toBeLessThan(120)
  })
})

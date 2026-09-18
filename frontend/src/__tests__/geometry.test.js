import { describe, expect, it } from 'vitest'
import { calculateObjectFit } from '../composables/useDetectionCanvas'

describe('bounding box coordinate transform', () => {
  it('calculates letterbox offsets for portrait containers', () => {
    const value = calculateObjectFit(360, 640, 640, 360, 'contain')
    expect(value.scale).toBeCloseTo(0.5625)
    expect(value.offsetY).toBeCloseTo(218.75)
  })
  it('calculates crop offsets for cover', () => {
    const value = calculateObjectFit(360, 640, 640, 360, 'cover')
    expect(value.offsetX).toBeLessThan(0)
  })
})

import './setup'
import { render, screen } from '@testing-library/vue'
import { describe, expect, it } from 'vitest'
import DetectionResult from '../components/DetectionResult.vue'

describe('detection result', () => {
  it('renders counts, confidence, result image and download', () => {
    render(DetectionResult, { props: { originalUrl: 'blob:original', result: {
      detected: true,
      fire_count: 1,
      smoke_count: 1,
      inference_ms: 12.4,
      result_image_url: '/api/results/x.jpg',
      detections: [{ class_name: 'fire', confidence: 0.92 }, { class_name: 'smoke', confidence: 0.81 }],
    } } })
    expect(screen.getByText('위험 요소 감지')).toBeTruthy()
    expect(screen.getByText('92%')).toBeTruthy()
    expect(screen.getByRole('link', { name: '결과 이미지 다운로드' }).getAttribute('href')).toBe('/api/results/x.jpg')
  })

  it('renders an explicit empty result', () => {
    render(DetectionResult, { props: { originalUrl: 'blob:original', result: {
      detected: false,
      fire_count: 0,
      smoke_count: 0,
      inference_ms: 5,
      result_image_url: '/api/results/empty.jpg',
      detections: [],
    } } })
    expect(screen.getByText('탐지 결과 없음')).toBeTruthy()
  })
})

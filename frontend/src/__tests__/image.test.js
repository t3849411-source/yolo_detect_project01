import './setup'
import { fireEvent, render, screen } from '@testing-library/vue'
import { describe, expect, it } from 'vitest'
import ImageDetector from '../components/ImageDetector.vue'

describe('image picker', () => {
  it('validates no selection and previews a selected photo', async () => {
    const { container } = render(ImageDetector)
    await fireEvent.click(screen.getByText('탐지 시작'))
    expect(screen.getByText(/먼저 분석할 사진/)).toBeTruthy()
    const input = container.querySelector('input[type=file]')
    await fireEvent.change(input, { target: { files: [new File(['x'], 'fire.jpg', { type: 'image/jpeg' })] } })
    expect(screen.getByAltText('선택한 사진 미리보기')).toBeTruthy()
    expect(input.hasAttribute('capture')).toBe(false)
  })

  it('rejects unsupported extensions and oversized files', async () => {
    const { container } = render(ImageDetector)
    const input = container.querySelector('input[type=file]')
    await fireEvent.change(input, { target: { files: [new File(['x'], 'fire.gif', { type: 'image/gif' })] } })
    expect(screen.getByText(/JPEG, PNG, WebP/)).toBeTruthy()
    const huge = new File([new Uint8Array(15 * 1024 * 1024 + 1)], 'huge.jpg', { type: 'image/jpeg' })
    await fireEvent.change(input, { target: { files: [huge] } })
    expect(screen.getByText(/15MB 이하/)).toBeTruthy()
  })
})

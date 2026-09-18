import './setup'
import { fireEvent, render, screen } from '@testing-library/vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App.vue'

describe('mode navigation', () => {
  beforeEach(() => vi.clearAllMocks())

  it('shows the requested title, warning, and both mode buttons', () => {
    render(App)
    expect(screen.getByText('AI 화염·연기 탐지')).toBeTruthy()
    expect(screen.getByText('카메라 실시간 탐지')).toBeTruthy()
    expect(screen.getByText('갤러리 사진 탐지')).toBeTruthy()
    expect(screen.getByText(/실제 화재경보기나 안전설비를 대체하지 않습니다/)).toBeTruthy()
  })

  it('navigates to camera and image modes and back', async () => {
    render(App)
    await fireEvent.click(screen.getByTestId('camera-mode'))
    expect(screen.getByText('실시간 카메라 탐지')).toBeTruthy()
    await fireEvent.click(screen.getByText(/첫 화면으로 돌아가기/))
    await fireEvent.click(screen.getByTestId('image-mode'))
    expect(screen.getByText('갤러리 사진 탐지')).toBeTruthy()
    await fireEvent.click(screen.getByText(/첫 화면으로 돌아가기/))
    expect(screen.getByTestId('camera-mode')).toBeTruthy()
  })
})

import './setup'
import { fireEvent, render, screen } from '@testing-library/vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import CameraDetector from '../components/CameraDetector.vue'

describe('camera controls', () => {
  beforeEach(() => {
    global.WebSocket = class {
      static OPEN = 1
      static CONNECTING = 0
      constructor() { this.readyState = 1; setTimeout(() => this.onopen?.(), 0) }
      close() { this.onclose?.() }
      send() {}
    }
  })

  it('requests the rear camera and stops every track', async () => {
    const stop = vi.fn()
    const getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [{ stop }] })
    Object.defineProperty(navigator, 'mediaDevices', { value: { getUserMedia }, configurable: true })
    render(CameraDetector)
    await fireEvent.click(screen.getByText('카메라 시작'))
    expect(getUserMedia).toHaveBeenCalledWith(expect.objectContaining({
      video: expect.objectContaining({ facingMode: { ideal: 'environment' } }),
      audio: false,
    }))
    await fireEvent.click(screen.getByText('카메라 중지'))
    expect(stop).toHaveBeenCalled()
  })

  it('shows a useful permission denial error', async () => {
    const error = Object.assign(new Error(), { name: 'NotAllowedError' })
    Object.defineProperty(navigator, 'mediaDevices', { value: { getUserMedia: vi.fn().mockRejectedValue(error) }, configurable: true })
    render(CameraDetector)
    await fireEvent.click(screen.getByText('카메라 시작'))
    expect(await screen.findByText(/카메라 권한이 거부/)).toBeTruthy()
  })
})

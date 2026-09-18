import { vi } from 'vitest'

global.ResizeObserver = class { observe() {} disconnect() {} }
Object.defineProperty(window, 'devicePixelRatio', { value: 1, configurable: true })
Object.defineProperty(window, 'isSecureContext', { value: true, configurable: true })
HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined)
HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
  setTransform: vi.fn(), clearRect: vi.fn(), strokeRect: vi.fn(), fillRect: vi.fn(), fillText: vi.fn(),
  measureText: vi.fn(() => ({ width: 40 })), drawImage: vi.fn(), save: vi.fn(), restore: vi.fn(), translate: vi.fn(), scale: vi.fn(),
}))
HTMLCanvasElement.prototype.toBlob = vi.fn((callback) => callback(new Blob(['frame'], { type: 'image/jpeg' })))
global.URL.createObjectURL = vi.fn(() => 'blob:preview')
global.URL.revokeObjectURL = vi.fn()

import { describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'
import { resolveWebSocketUrl, useDetectionSocket } from '../composables/useDetectionSocket'

describe('websocket transport', () => {
  it('uses wss on https', () => expect(resolveWebSocketUrl()).toBe('wss://example.test/ws/detect'))
  it('prevents a second frame while one is in flight', () => {
    class FakeSocket { static OPEN=1; static CONNECTING=0; constructor(){ this.readyState=1; FakeSocket.instance=this } send=vi.fn(); close(){} }
    global.WebSocket = FakeSocket
    const socket = useDetectionSocket({ confidence: ref(0.3), onResult: vi.fn() })
    socket.connect(); FakeSocket.instance.onopen()
    expect(socket.send(new Blob(['a']))).toBe(true)
    expect(socket.send(new Blob(['b']))).toBe(false)
    expect(FakeSocket.instance.send).toHaveBeenCalledTimes(1)
  })
})

import { computed, onBeforeUnmount, ref } from 'vue'

export function resolveWebSocketUrl(path = '/ws/detect') {
  const configured = import.meta.env.VITE_WS_BASE_URL?.replace(/\/$/, '')
  if (configured) return `${configured}${path}`
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${wsProtocol}//${window.location.host}${path}`
}

export function useDetectionSocket({ onResult, confidence }) {
  const status = ref('disconnected')
  const inFlight = ref(false)
  const roundTripMs = ref(0)
  const retries = ref(0)
  let socket = null
  let sentAt = 0
  let intentionalClose = false
  let reconnectTimer = null

  function connect() {
    if (socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(socket.readyState)) return
    clearTimeout(reconnectTimer)
    intentionalClose = false
    status.value = 'connecting'
    const url = new URL(resolveWebSocketUrl())
    url.searchParams.set('confidence', confidence.value)
    socket = new WebSocket(url)
    socket.binaryType = 'arraybuffer'
    socket.onopen = () => {
      status.value = 'connected'
      retries.value = 0
    }
    socket.onmessage = (event) => {
      roundTripMs.value = sentAt ? performance.now() - sentAt : 0
      inFlight.value = false
      try {
        onResult(JSON.parse(event.data))
      } catch {
        onResult({ error: '서버 응답을 해석할 수 없습니다.' })
      }
    }
    socket.onerror = () => { status.value = 'error' }
    socket.onclose = () => {
      status.value = 'disconnected'
      inFlight.value = false
      socket = null
      if (!intentionalClose && retries.value < 3) {
        const delay = 700 * 2 ** retries.value++
        reconnectTimer = window.setTimeout(connect, delay)
      }
    }
  }

  function send(blob) {
    if (!blob || inFlight.value || socket?.readyState !== WebSocket.OPEN) return false
    inFlight.value = true
    sentAt = performance.now()
    socket.send(blob)
    return true
  }

  function disconnect() {
    intentionalClose = true
    clearTimeout(reconnectTimer)
    reconnectTimer = null
    const current = socket
    socket = null
    current?.close(1000, 'client stop')
    status.value = 'disconnected'
    inFlight.value = false
  }

  const connected = computed(() => status.value === 'connected')
  onBeforeUnmount(disconnect)
  return { status, connected, inFlight, roundTripMs, connect, send, disconnect }
}

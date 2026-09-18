<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import DetectionCanvas from './DetectionCanvas.vue'
import SettingsPanel from './SettingsPanel.vue'
import { useCamera } from '../composables/useCamera'
import { useDetectionSocket } from '../composables/useDetectionSocket'

const emit = defineEmits(['back'])
const video = ref(null)
const capture = ref(null)
const confidence = ref(0.35)
const targetFps = ref(5)
const detections = ref([])
const result = ref({ image_width: 640, image_height: 360, fire_count: 0, smoke_count: 0, inference_ms: 0 })
const actualFps = ref(0)
const vibration = ref(false)
const soundEnabled = ref(false)
const warningUntil = ref(0)
const now = ref(Date.now())
let audioContext
let timer
let clock
let lastResultAt = 0
let lastAlertAt = 0
let running = false

const camera = useCamera(video)
const socket = useDetectionSocket({ confidence, onResult })
const warningActive = computed(() => now.value < warningUntil.value)
const connectionText = computed(() => ({ connected: '연결됨', connecting: '연결 중', disconnected: '연결 끊김', error: '연결 오류' }[socket.status.value]))
const warningText = computed(() => {
  const names = []
  if (result.value.fire_count) names.push(`화염 ${result.value.fire_count}`)
  if (result.value.smoke_count) names.push(`연기 ${result.value.smoke_count}`)
  const highest = Math.max(0, ...detections.value.map((item) => item.confidence))
  return `${names.join(' · ')} · 최고 신뢰도 ${Math.round(highest * 100)}%`
})

function onResult(payload) {
  if (payload.error) {
    camera.error.value = payload.error
    return
  }
  detections.value = payload.detections || []
  result.value = payload
  const time = performance.now()
  actualFps.value = lastResultAt ? 1000 / (time - lastResultAt) : 0
  lastResultAt = time
  if (payload.fire_count || payload.smoke_count) {
    warningUntil.value = Date.now() + 1800
    alertUser()
  }
}

function alertUser() {
  if (Date.now() - lastAlertAt < 5000) return
  lastAlertAt = Date.now()
  if (vibration.value && navigator.vibrate) navigator.vibrate([180, 100, 180])
  if (soundEnabled.value && audioContext) {
    const oscillator = audioContext.createOscillator()
    const gain = audioContext.createGain()
    oscillator.frequency.value = 740
    gain.gain.value = 0.08
    oscillator.connect(gain).connect(audioContext.destination)
    oscillator.start()
    oscillator.stop(audioContext.currentTime + 0.18)
  }
}

async function enableSound() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext
  if (!AudioContextClass) {
    camera.error.value = '이 브라우저는 경고음을 지원하지 않습니다.'
    return
  }
  audioContext ||= new AudioContextClass()
  await audioContext.resume()
  soundEnabled.value = true
}

function frameBlob() {
  return new Promise((resolve) => {
    const canvas = capture.value
    const source = video.value
    if (!canvas || !source?.videoWidth) return resolve(null)
    canvas.width = 640
    canvas.height = 360
    const context = canvas.getContext('2d')
    const sourceRatio = source.videoWidth / source.videoHeight
    const targetRatio = 640 / 360
    let sx = 0; let sy = 0; let sw = source.videoWidth; let sh = source.videoHeight
    if (sourceRatio > targetRatio) {
      sw = source.videoHeight * targetRatio
      sx = (source.videoWidth - sw) / 2
    } else {
      sh = source.videoWidth / targetRatio
      sy = (source.videoHeight - sh) / 2
    }
    if (camera.facingMode.value === 'user') {
      context.save()
      context.translate(640, 0)
      context.scale(-1, 1)
      context.drawImage(source, sx, sy, sw, sh, 0, 0, 640, 360)
      context.restore()
    } else {
      context.drawImage(source, sx, sy, sw, sh, 0, 0, 640, 360)
    }
    canvas.toBlob(resolve, 'image/jpeg', 0.7)
  })
}

async function tick() {
  if (!running) return
  if (camera.active.value && !document.hidden && socket.connected.value && !socket.inFlight.value) {
    socket.send(await frameBlob())
  }
  if (!running) return
  const networkFloor = Math.max(1000 / targetFps.value, socket.roundTripMs.value * 1.15)
  timer = window.setTimeout(tick, networkFloor)
}

async function start() {
  if (await camera.start()) {
    running = true
    socket.connect()
    clearTimeout(timer)
    tick()
  }
}

function stop() {
  running = false
  clearTimeout(timer)
  camera.stop()
  socket.disconnect()
  detections.value = []
  result.value = { image_width: 640, image_height: 360, fire_count: 0, smoke_count: 0, inference_ms: 0 }
  actualFps.value = 0
  warningUntil.value = 0
}

async function switchCamera() {
  await camera.switchCamera()
}

function goBack() {
  stop()
  emit('back')
}

function visibilityChanged() {
  clearTimeout(timer)
  if (!document.hidden && running && camera.active.value) {
    socket.connect()
    tick()
  }
}

watch(confidence, () => {
  if (running && camera.active.value) {
    socket.disconnect()
    socket.connect()
  }
})
onMounted(() => {
  document.addEventListener('visibilitychange', visibilityChanged)
  clock = window.setInterval(() => { now.value = Date.now() }, 250)
})
onBeforeUnmount(() => {
  stop()
  clearInterval(clock)
  document.removeEventListener('visibilitychange', visibilityChanged)
  audioContext?.close()
})
</script>

<template>
  <section class="detector-view">
    <button class="back-button" type="button" @click="goBack">← 첫 화면으로 돌아가기</button>
    <div v-if="warningActive" class="warning-banner" role="alert"><b>위험 요소 감지</b><span>{{ warningText }}</span></div>
    <div class="view-heading">
      <div><span class="kicker">LIVE ANALYSIS</span><h2>실시간 카메라 탐지</h2></div>
      <span class="connection" :class="socket.status.value">● {{ connectionText }}</span>
    </div>
    <div class="camera-stage">
      <video ref="video" autoplay muted playsinline :class="{ mirrored: camera.facingMode.value === 'user' }"></video>
      <DetectionCanvas :detections="detections" :source-width="result.image_width" :source-height="result.image_height" fit="cover" />
      <div v-if="!camera.active.value" class="camera-placeholder"><span>●</span><p>카메라를 시작하면 이곳에 영상이 표시됩니다.</p></div>
      <canvas ref="capture" hidden></canvas>
    </div>
    <p v-if="camera.error.value" class="error-message" role="alert">{{ camera.error.value }}</p>
    <div class="metrics-grid">
      <div><small>화염</small><b class="fire-text">{{ result.fire_count }}</b></div>
      <div><small>연기</small><b class="smoke-text">{{ result.smoke_count }}</b></div>
      <div><small>실제 FPS</small><b>{{ actualFps.toFixed(1) }}</b></div>
      <div><small>서버 / 왕복</small><b>{{ result.inference_ms.toFixed(0) }} / {{ socket.roundTripMs.value.toFixed(0) }} ms</b></div>
    </div>
    <div class="camera-actions">
      <button class="primary-button" type="button" :disabled="camera.active.value" @click="start">카메라 시작</button>
      <button class="secondary-button" type="button" :disabled="!camera.active.value" @click="stop">카메라 중지</button>
      <button class="secondary-button" type="button" :disabled="!camera.active.value" @click="switchCamera">전면·후면 전환</button>
    </div>
    <SettingsPanel v-model:confidence="confidence" v-model:fps="targetFps" v-model:vibration="vibration" :sound="soundEnabled" @enable-sound="enableSound" />
  </section>
</template>

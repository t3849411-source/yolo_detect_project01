import { onBeforeUnmount, ref } from 'vue'

export function cameraErrorMessage(error) {
  const messages = {
    NotAllowedError: '카메라 권한이 거부되었습니다. 브라우저 설정에서 권한을 허용해 주세요.',
    NotFoundError: '사용 가능한 카메라를 찾을 수 없습니다.',
    NotReadableError: '다른 앱이 카메라를 사용 중입니다.',
    OverconstrainedError: '요청한 카메라 조건을 지원하지 않습니다.',
    SecurityError: '카메라는 HTTPS 보안 연결에서만 사용할 수 있습니다.',
  }
  return messages[error?.name] || '카메라를 시작할 수 없습니다.'
}

export function useCamera(videoRef) {
  const stream = ref(null)
  const facingMode = ref('environment')
  const error = ref('')
  const active = ref(false)

  async function start() {
    error.value = ''
    if (!window.isSecureContext && !['localhost', '127.0.0.1'].includes(window.location.hostname)) {
      error.value = '스마트폰 카메라는 HTTPS 보안 연결이 필요합니다.'
      return false
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      error.value = '이 브라우저는 카메라 API를 지원하지 않습니다.'
      return false
    }
    stop()
    try {
      stream.value = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facingMode.value },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      })
      if (videoRef.value) {
        videoRef.value.srcObject = stream.value
        await videoRef.value.play()
      }
      active.value = true
      return true
    } catch (reason) {
      error.value = cameraErrorMessage(reason)
      active.value = false
      return false
    }
  }

  function stop() {
    stream.value?.getTracks().forEach((track) => track.stop())
    stream.value = null
    if (videoRef.value) videoRef.value.srcObject = null
    active.value = false
  }

  async function switchCamera() {
    facingMode.value = facingMode.value === 'environment' ? 'user' : 'environment'
    return start()
  }

  onBeforeUnmount(stop)
  return { stream, facingMode, error, active, start, stop, switchCamera }
}

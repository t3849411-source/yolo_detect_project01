const configuredBase = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
export const apiUrl = (path) => `${configuredBase}${path}`

function errorMessage(status, payload) {
  if (payload?.detail) return payload.detail
  if (status === 413) return '파일 크기가 허용 한도를 초과했습니다.'
  if (status === 415) return '지원하지 않는 이미지 형식입니다.'
  if (status === 503) return '서버에서 모델을 사용할 수 없습니다.'
  return '서버에서 이미지를 처리하지 못했습니다.'
}

export function uploadImage(file, confidence, iou, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    const form = new FormData()
    form.append('file', file, file.name)
    form.append('confidence', String(confidence))
    form.append('iou', String(iou))
    request.open('POST', apiUrl('/api/detect/image'))
    request.timeout = 120_000
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100))
    }
    request.onerror = () => reject(new Error('네트워크 연결을 확인해 주세요.'))
    request.ontimeout = () => reject(new Error('서버 응답 시간이 초과되었습니다.'))
    request.onload = () => {
      let payload
      try { payload = JSON.parse(request.responseText) } catch { payload = {} }
      if (request.status >= 200 && request.status < 300) resolve(payload)
      else reject(new Error(errorMessage(request.status, payload)))
    }
    request.send(form)
  })
}

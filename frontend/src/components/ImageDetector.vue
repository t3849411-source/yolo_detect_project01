<script setup>
import { onBeforeUnmount, ref } from 'vue'
import DetectionResult from './DetectionResult.vue'
import { uploadImage } from '../services/api'

defineEmits(['back'])
const file = ref(null)
const preview = ref('')
const confidence = ref(0.35)
const iou = ref(0.45)
const progress = ref(0)
const busy = ref(false)
const error = ref('')
const result = ref(null)
const input = ref(null)
const allowedMime = new Set(['image/jpeg', 'image/png', 'image/webp'])
const allowedExtensions = new Set(['jpg', 'jpeg', 'png', 'webp'])
const maxBytes = 15 * 1024 * 1024

function clearSelection() {
  file.value = null
  result.value = null
  progress.value = 0
  if (preview.value) URL.revokeObjectURL(preview.value)
  preview.value = ''
  if (input.value) input.value.value = ''
}

function choose(selected) {
  error.value = ''
  result.value = null
  const candidate = selected?.[0]
  if (!candidate) return
  const extension = candidate.name.split('.').pop()?.toLowerCase()
  if (!allowedMime.has(candidate.type) || !allowedExtensions.has(extension)) {
    clearSelection()
    error.value = 'JPEG, PNG, WebP 사진만 선택할 수 있습니다.'
    return
  }
  if (candidate.size === 0) {
    clearSelection()
    error.value = '빈 파일은 선택할 수 없습니다.'
    return
  }
  if (candidate.size > maxBytes) {
    clearSelection()
    error.value = '사진 크기는 15MB 이하여야 합니다.'
    return
  }
  if (preview.value) URL.revokeObjectURL(preview.value)
  file.value = candidate
  preview.value = URL.createObjectURL(candidate)
  // 같은 파일을 연속으로 선택해도 change 이벤트가 다시 발생하도록 입력값을 비운다.
  if (input.value) input.value.value = ''
}

async function resizedFile(source) {
  if (!window.createImageBitmap) return source
  let bitmap
  try {
    bitmap = await createImageBitmap(source, { imageOrientation: 'from-image' })
  } catch {
    throw new Error('손상된 이미지이거나 브라우저가 읽을 수 없는 파일입니다.')
  }
  const limit = 2048
  const scale = Math.min(1, limit / Math.max(bitmap.width, bitmap.height))
  if (scale === 1) {
    bitmap.close()
    return source
  }
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(bitmap.width * scale)
  canvas.height = Math.round(bitmap.height * scale)
  canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height)
  bitmap.close()
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9))
  if (!blob) throw new Error('이미지 크기를 조정하지 못했습니다.')
  return new File([blob], `${source.name.replace(/\.[^.]+$/, '')}-resized.jpg`, { type: 'image/jpeg' })
}

async function detect() {
  if (!file.value) {
    error.value = '먼저 분석할 사진을 선택해 주세요.'
    return
  }
  busy.value = true
  error.value = ''
  progress.value = 0
  result.value = null
  try {
    const upload = await resizedFile(file.value)
    result.value = await uploadImage(upload, confidence.value, iou.value, (value) => { progress.value = value })
    progress.value = 100
  } catch (reason) {
    error.value = reason?.message || '사진 분석 중 오류가 발생했습니다.'
  } finally {
    busy.value = false
  }
}

function reset() {
  error.value = ''
  clearSelection()
}

function drop(event) {
  choose(event.dataTransfer.files)
}

onBeforeUnmount(() => {
  if (preview.value) URL.revokeObjectURL(preview.value)
})
</script>

<template>
  <section class="detector-view">
    <button class="back-button" type="button" @click="$emit('back')">← 첫 화면으로 돌아가기</button>
    <div class="view-heading"><div><span class="kicker">STILL IMAGE ANALYSIS</span><h2>갤러리 사진 탐지</h2></div></div>
    <div v-if="!result" class="upload-layout">
      <label class="drop-zone" @dragover.prevent @drop.prevent="drop">
        <input ref="input" type="file" accept="image/jpeg,image/png,image/webp" @change="choose($event.target.files)" />
        <template v-if="preview"><img :src="preview" alt="선택한 사진 미리보기" /></template>
        <template v-else>
          <span aria-hidden="true">＋</span>
          <strong>사진 선택</strong>
          <small>눌러서 갤러리 사진을 선택하거나 파일을 놓으세요<br />JPEG · PNG · WebP / 최대 15MB</small>
        </template>
      </label>
      <div class="analysis-controls">
        <label>Confidence <output>{{ confidence.toFixed(2) }}</output><input v-model.number="confidence" type="range" min="0" max="1" step="0.05" /></label>
        <label>IoU <output>{{ iou.toFixed(2) }}</output><input v-model.number="iou" type="range" min="0" max="1" step="0.05" /></label>
        <button class="primary-button" type="button" :disabled="busy" @click="detect">{{ busy ? `분석 중 ${progress}%` : '탐지 시작' }}</button>
        <div v-if="busy" class="progress-track" role="progressbar" :aria-valuenow="progress" aria-valuemin="0" aria-valuemax="100"><i :style="{ width: `${progress}%` }"></i></div>
      </div>
    </div>
    <p v-if="error" class="error-message" role="alert">{{ error }}</p>
    <template v-if="result">
      <DetectionResult :result="result" :original-url="preview" />
      <button class="secondary-button full-button" type="button" @click="reset">다른 사진 선택</button>
    </template>
  </section>
</template>

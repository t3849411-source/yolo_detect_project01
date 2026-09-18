<script setup>
import { computed } from 'vue'
import { apiUrl } from '../services/api'

const props = defineProps({ result: { type: Object, required: true }, originalUrl: String })
const resultUrl = computed(() => apiUrl(props.result.result_image_url))
</script>

<template>
  <section class="result-panel" aria-live="polite">
    <div v-if="result.detected" class="warning-banner" role="alert">
      <b>위험 요소 감지</b><span>화염 {{ result.fire_count }} · 연기 {{ result.smoke_count }}</span>
    </div>
    <div v-else class="clear-banner">탐지된 화염 또는 연기가 없습니다.</div>
    <div class="comparison-grid">
      <figure><figcaption>원본 사진</figcaption><img :src="originalUrl" alt="업로드한 원본" /></figure>
      <figure><figcaption>탐지 결과</figcaption><img :src="resultUrl" alt="경계 상자가 표시된 탐지 결과" /></figure>
    </div>
    <div class="metrics-grid">
      <div><small>화염</small><b class="fire-text">{{ result.fire_count }}</b></div>
      <div><small>연기</small><b class="smoke-text">{{ result.smoke_count }}</b></div>
      <div><small>추론 시간</small><b>{{ result.inference_ms.toFixed(1) }} ms</b></div>
    </div>
    <div v-if="result.detections.length" class="confidence-list" aria-label="탐지별 신뢰도">
      <p v-for="(item, index) in result.detections" :key="index"><span>{{ item.class_name }}</span><b>{{ Math.round(item.confidence * 100) }}%</b></p>
    </div>
    <p v-else class="empty-result">탐지 결과 없음</p>
    <a class="primary-button download-button" :href="resultUrl" download>결과 이미지 다운로드</a>
  </section>
</template>

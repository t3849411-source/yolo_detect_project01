<script setup>
import { ref } from 'vue'
import CameraDetector from './components/CameraDetector.vue'
import ImageDetector from './components/ImageDetector.vue'
import ModeSelector from './components/ModeSelector.vue'

const mode = ref('home')
</script>

<template>
  <main class="app-shell">
    <header class="brand-header">
      <div class="brand-mark" aria-hidden="true">F</div>
      <div>
        <p class="eyebrow">VISION SAFETY LAB</p>
        <h1>AI 화염·연기 탐지</h1>
      </div>
      <span class="system-dot"><i></i> 시스템 준비</span>
    </header>

    <ModeSelector v-if="mode === 'home'" @select="mode = $event" />
    <CameraDetector v-else-if="mode === 'camera'" @back="mode = 'home'" />
    <ImageDetector v-else @back="mode = 'home'" />

    <footer class="safety-note">
      <span aria-hidden="true">!</span>
      <p>이 서비스는 학습 및 보조 탐지용이며<br />실제 화재경보기나 안전설비를 대체하지 않습니다.</p>
    </footer>
  </main>
</template>

<script setup>
defineEmits(['select'])
</script>

<template>
  <section class="home-view">
    <div class="hero-copy">
      <span class="kicker">SELECT DETECTION MODE</span>
      <h2>어떤 방식으로<br /><em>탐지할까요?</em></h2>
      <p>사진과 카메라 프레임은 연결된 PC의 GPU에서 분석됩니다.</p>
    </div>
    <div class="mode-grid">
      <button class="mode-card fire-card" data-testid="camera-mode" @click="$emit('select', 'camera')">
        <span class="mode-icon" aria-hidden="true">●</span>
        <span class="mode-number">01 / LIVE</span>
        <strong>카메라 실시간 탐지</strong>
        <small>후면 카메라 영상을 전송하고<br />화염·연기를 실시간으로 분석합니다.</small>
        <b>시작하기 <span>→</span></b>
      </button>
      <button class="mode-card smoke-card" data-testid="image-mode" @click="$emit('select', 'image')">
        <span class="mode-icon" aria-hidden="true">◆</span>
        <span class="mode-number">02 / IMAGE</span>
        <strong>갤러리 사진 탐지</strong>
        <small>사진 한 장을 선택해 정밀 분석하고<br />결과 이미지를 저장합니다.</small>
        <b>사진 선택 <span>→</span></b>
      </button>
    </div>
    <div class="feature-strip">
      <span>GPU 추론</span><span>FIRE</span><span>SMOKE</span><span>실시간 좌표</span>
    </div>
  </section>
</template>

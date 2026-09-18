<script setup>
defineProps({ confidence: Number, fps: Number, vibration: Boolean, sound: Boolean })
defineEmits(['update:confidence', 'update:fps', 'update:vibration', 'enable-sound'])
</script>

<template>
  <section class="settings-panel" aria-label="실시간 탐지 설정">
    <label>Confidence <output>{{ confidence.toFixed(2) }}</output>
      <input type="range" min="0.1" max="0.9" step="0.05" :value="confidence" @input="$emit('update:confidence', +$event.target.value)" />
    </label>
    <label>전송 FPS
      <select :value="fps" @change="$emit('update:fps', +$event.target.value)">
        <option v-for="item in [1, 3, 5, 10]" :key="item" :value="item">{{ item }}</option>
      </select>
    </label>
    <label class="toggle"><input type="checkbox" :checked="vibration" @change="$emit('update:vibration', $event.target.checked)" /> 진동 경고</label>
    <button class="text-button" type="button" @click="$emit('enable-sound')">{{ sound ? '소리 경고 켜짐' : '소리 경고 활성화' }}</button>
  </section>
</template>

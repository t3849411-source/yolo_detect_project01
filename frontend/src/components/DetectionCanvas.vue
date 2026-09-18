<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { drawDetections } from '../composables/useDetectionCanvas'

const props = defineProps({
  detections: { type: Array, default: () => [] },
  sourceWidth: { type: Number, default: 640 },
  sourceHeight: { type: Number, default: 360 },
  fit: { type: String, default: 'contain' },
})
const canvas = ref(null)
let observer
const redraw = () => drawDetections(canvas.value, props.detections, props.sourceWidth, props.sourceHeight, props.fit)
watch(() => [props.detections, props.sourceWidth, props.sourceHeight, props.fit], redraw, { deep: true })
onMounted(() => {
  observer = new ResizeObserver(redraw)
  observer.observe(canvas.value)
  redraw()
})
onBeforeUnmount(() => observer?.disconnect())
</script>

<template><canvas ref="canvas" class="detection-canvas" aria-label="탐지 경계 상자"></canvas></template>

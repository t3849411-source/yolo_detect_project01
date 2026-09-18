export function calculateObjectFit(containerWidth, containerHeight, sourceWidth, sourceHeight, fit = 'contain') {
  if (!containerWidth || !containerHeight || !sourceWidth || !sourceHeight) {
    return { scale: 1, offsetX: 0, offsetY: 0 }
  }
  const scale = fit === 'cover'
    ? Math.max(containerWidth / sourceWidth, containerHeight / sourceHeight)
    : Math.min(containerWidth / sourceWidth, containerHeight / sourceHeight)
  return {
    scale,
    offsetX: (containerWidth - sourceWidth * scale) / 2,
    offsetY: (containerHeight - sourceHeight * scale) / 2,
  }
}

export function drawDetections(canvas, detections, sourceWidth, sourceHeight, fit = 'contain') {
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const pixelRatio = window.devicePixelRatio || 1
  canvas.width = Math.round(rect.width * pixelRatio)
  canvas.height = Math.round(rect.height * pixelRatio)
  const context = canvas.getContext('2d')
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
  context.clearRect(0, 0, rect.width, rect.height)
  const { scale, offsetX, offsetY } = calculateObjectFit(rect.width, rect.height, sourceWidth, sourceHeight, fit)
  context.font = '600 13px system-ui'
  context.lineWidth = 3
  detections.forEach((item) => {
    const color = item.class_name.toLowerCase() === 'fire' ? '#ff3b30' : '#ffb000'
    const x = offsetX + item.x1 * scale
    const y = offsetY + item.y1 * scale
    const width = (item.x2 - item.x1) * scale
    const height = (item.y2 - item.y1) * scale
    context.strokeStyle = color
    context.strokeRect(x, y, width, height)
    const label = `${item.class_name.toUpperCase()} ${Math.round(item.confidence * 100)}%`
    const labelWidth = context.measureText(label).width + 12
    const labelY = Math.max(0, y - 25)
    context.fillStyle = color
    context.fillRect(x, labelY, labelWidth, 25)
    context.fillStyle = '#080b10'
    context.fillText(label, x + 6, labelY + 17)
  })
}

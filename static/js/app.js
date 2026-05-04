function initChart(id, config) {
  const canvas = document.getElementById(id)
  if (!canvas || !window.Chart) return
  new Chart(canvas, config)
}

document.addEventListener("DOMContentLoaded", () => {
  const audioInput = document.getElementById("audio_file")
  const selectedFile = document.getElementById("selectedAudioFile")
  const uploadZone = audioInput?.closest(".upload-zone-ingest")

  if (audioInput && selectedFile && uploadZone) {
    const selectedFileText = selectedFile.querySelector(".selected-file-text")
    const uploadTitle = uploadZone.querySelector(".upload-title")

    audioInput.addEventListener("change", () => {
      const file = audioInput.files?.[0]
      if (!file) {
        uploadZone.classList.remove("has-file")
        selectedFile.classList.add("d-none")
        if (uploadTitle) uploadTitle.textContent = "Drag and drop audio files"
        if (selectedFileText) selectedFileText.textContent = "No file selected"
        return
      }

      const fileSizeMb = file.size ? ` (${(file.size / (1024 * 1024)).toFixed(2)} MB)` : ""
      uploadZone.classList.add("has-file")
      selectedFile.classList.remove("d-none")
      if (uploadTitle) uploadTitle.textContent = "Audio file selected"
      if (selectedFileText) selectedFileText.textContent = `${file.name}${fileSizeMb}`
    })
  }

  if (window.vishhunterCharts) {
    initChart("riskChart", {
      type: "pie",
      data: {
        labels: ["Low Risk", "Medium Risk", "High Risk"],
        datasets: [
          {
            data: window.vishhunterCharts.riskDistribution,
            backgroundColor: ["#2daa43", "#e7a900", "#dc143c"],
            borderColor: "#08111c",
            borderWidth: 3,
          },
        ],
      },
      options: { plugins: { legend: { labels: { color: "#f8fafc" } } } },
    })

    initChart("trendChart", {
      type: "line",
      data: {
        labels: window.vishhunterCharts.weeklyTrendLabels,
        datasets: [
          {
            data: window.vishhunterCharts.weeklyTrendValues,
            borderColor: "#16a8ff",
            backgroundColor: "rgba(22, 168, 255, 0.15)",
            tension: 0.35,
            fill: false,
          },
        ],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#8ca0b8" }, grid: { color: "rgba(148,163,184,0.12)" } },
          y: { ticks: { color: "#8ca0b8" }, grid: { color: "rgba(148,163,184,0.12)" } },
        },
      },
    })
  }

  if (window.vishhunterReports) {
    initChart("monthlyChart", {
      type: "bar",
      data: {
        labels: window.vishhunterReports.monthlyLabels,
        datasets: [{ data: window.vishhunterReports.monthlyTotals, backgroundColor: "#16a8ff" }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#8ca0b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
          y: { ticks: { color: "#8ca0b8" }, grid: { color: "rgba(148,163,184,0.12)" } },
        },
      },
    })

    initChart("reportRiskChart", {
      type: "doughnut",
      data: {
        labels: ["Low", "Medium", "High"],
        datasets: [
          {
            data: window.vishhunterReports.riskCounts,
            backgroundColor: ["#2daa43", "#e7a900", "#dc143c"],
          },
        ],
      },
      options: { plugins: { legend: { labels: { color: "#f8fafc" } } } },
    })
  }

  const waveform = document.getElementById("waveform")
  if (waveform && window.WaveSurfer) {
    const audioUrl = waveform.dataset.audioUrl
    if (audioUrl) {
      const ws = WaveSurfer.create({
        container: waveform,
        waveColor: "#5b6d82",
        progressColor: "#16a8ff",
        cursorColor: "#f8fafc",
        barWidth: 3,
        barRadius: 4,
        height: 120,
      })
      ws.load(audioUrl)
    }
  }
})

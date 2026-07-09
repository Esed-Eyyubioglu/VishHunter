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

  const notificationConfig = window.vishhunterNotificationConfig
  const toastStack = document.getElementById("notificationToastStack")

  if (notificationConfig && toastStack) {
    const seenKey = `vishhunterSeenNotifications:${notificationConfig.userId}`
    const storedIds = JSON.parse(window.localStorage.getItem(seenKey) || "[]")
    const seenIds = new Set([...storedIds, ...(notificationConfig.initialIds || [])])
    window.localStorage.setItem(seenKey, JSON.stringify([...seenIds].slice(-80)))

    const updateNotificationBadges = (count) => {
      if (count > 0 && !document.querySelector(".notification-count")) {
        const trigger = document.querySelector(".notification-trigger")
        if (trigger) {
          const countBadge = document.createElement("span")
          countBadge.className = "notification-count"
          trigger.appendChild(countBadge)
        }
      }
      if (count > 0 && !document.querySelector(".nav-badge")) {
        const navLink = document.querySelector('.nav-link[href="/notifications"]')
        if (navLink) {
          const navBadge = document.createElement("span")
          navBadge.className = "nav-badge"
          navLink.appendChild(navBadge)
        }
      }
      document.querySelectorAll(".notification-count, .nav-badge").forEach((badge) => {
        if (count > 0) {
          badge.textContent = count
          badge.classList.remove("d-none")
        } else {
          badge.classList.add("d-none")
        }
      })
    }

    const showNotificationToast = (notification) => {
      const toast = document.createElement("a")
      toast.className = `notification-toast notification-toast-${notification.severity || "info"}`
      toast.href = notification.url || "/notifications"

      const icon = document.createElement("span")
      icon.className = "material-symbols-outlined notification-toast-icon"
      icon.textContent = notification.severity === "danger" ? "error" : notification.severity === "warning" ? "warning" : notification.severity === "success" ? "check_circle" : "notifications"

      const body = document.createElement("span")
      body.className = "notification-toast-body"

      const title = document.createElement("strong")
      title.textContent = notification.title || "New notification"

      const message = document.createElement("small")
      message.textContent = notification.message || ""

      const time = document.createElement("em")
      time.textContent = notification.created_at || ""

      const close = document.createElement("button")
      close.type = "button"
      close.className = "notification-toast-close"
      close.setAttribute("aria-label", "Dismiss notification")
      close.innerHTML = "&times;"
      close.addEventListener("click", (event) => {
        event.preventDefault()
        event.stopPropagation()
        toast.remove()
      })

      body.append(title, message, time)
      toast.append(icon, body, close)
      toastStack.prepend(toast)

      window.setTimeout(() => {
        toast.classList.add("is-hiding")
        window.setTimeout(() => toast.remove(), 220)
      }, 7000)
    }

    const pollNotifications = async () => {
      try {
        const response = await fetch(notificationConfig.pollUrl, {
          headers: { Accept: "application/json" },
          cache: "no-store",
        })
        if (!response.ok) return
        const data = await response.json()
        updateNotificationBadges(data.unread_count || 0)

        const freshNotifications = (data.notifications || []).filter((notification) => !seenIds.has(notification.id))
        freshNotifications.reverse().forEach((notification) => {
          seenIds.add(notification.id)
          showNotificationToast(notification)
        })

        if (freshNotifications.length) {
          window.localStorage.setItem(seenKey, JSON.stringify([...seenIds].slice(-80)))
        }
      } catch (error) {
        console.warn("Notification polling failed", error)
      }
    }

    window.setInterval(pollNotifications, 10000)
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
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#f8fafc" } } },
      },
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
        responsive: true,
        maintainAspectRatio: false,
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
        responsive: true,
        maintainAspectRatio: false,
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
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#f8fafc" } } },
      },
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

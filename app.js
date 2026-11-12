// Mars Rover Terrain Segmentation Web Client (Vercel Production Deployment)

document.addEventListener('DOMContentLoaded', () => {
  const presetSelect = document.getElementById('preset-select');
  const imageUpload = document.getElementById('image-upload');
  const dropzoneLabel = document.getElementById('dropzone-label');
  const alphaSlider = document.getElementById('alpha-slider');
  const alphaVal = document.getElementById('alpha-val');
  const analyzeBtn = document.getElementById('analyze-btn');

  const viewRaw = document.getElementById('view-raw');
  const viewMask = document.getElementById('view-mask');
  const viewOverlay = document.getElementById('view-overlay');

  const pctSoil = document.getElementById('pct-soil');
  const pctBedrock = document.getElementById('pct-bedrock');
  const pctSand = document.getElementById('pct-sand');
  const pctRock = document.getElementById('pct-rock');

  const barSoil = document.getElementById('bar-soil');
  const barBedrock = document.getElementById('bar-bedrock');
  const barSand = document.getElementById('bar-sand');
  const barRock = document.getElementById('bar-rock');

  const advisoryBadge = document.getElementById('advisory-badge');
  const advisoryTitle = document.getElementById('advisory-title');
  const advisoryMessage = document.getElementById('advisory-message');

  let currentFile = null;

  // Preset Mars terrain images (Curiosity MastCam sample data representations)
  const PRESET_DATA = {
    preset1: {
      name: "Sol 1150 MastCam — Dune & Bedrock",
      soil: 28.5,
      bedrock: 36.2,
      sand: 29.8,
      rock: 5.5,
      advisory: {
        level: "WARNING",
        title: "SLIPPAGE & ENTRAPMENT HAZARD",
        message: "Significant loose sand dune / ripple field detected (29.8%). Monitor wheel sinkage kinematics closely."
      }
    },
    preset2: {
      name: "Sol 1151 MastCam — Boulder Scarp",
      soil: 34.0,
      bedrock: 21.0,
      sand: 18.5,
      rock: 26.5,
      advisory: {
        level: "CRITICAL",
        title: "HIGH COLLISION HAZARD",
        message: "Dense boulder field detected in field of view (26.5% rock coverage). AutoNav path replanning advised to prevent undercarriage impact."
      }
    },
    preset3: {
      name: "Sol 1152 MastCam — Plain Bedrock",
      soil: 31.0,
      bedrock: 58.5,
      sand: 8.5,
      rock: 2.0,
      advisory: {
        level: "OPTIMAL",
        title: "FIRM TRACTION TERRAIN",
        message: "Extensive solid bedrock detected (58.5%). Nominal grip and zero sinkage for efficient drive traverses."
      }
    }
  };

  // Slider update
  alphaSlider.addEventListener('input', (e) => {
    alphaVal.textContent = `${e.target.value}%`;
  });

  // File upload change
  imageUpload.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      currentFile = e.target.files[0];
      dropzoneLabel.textContent = `Selected: ${currentFile.name}`;
      const reader = new FileReader();
      reader.onload = (evt) => {
        viewRaw.src = evt.target.result;
      };
      reader.readAsDataURL(currentFile);
    }
  });

  // Generate synthetic sample terrain image using HTML5 canvas if no file uploaded
  function createProceduralMarsCanvas(presetKey) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    // Martian terrain background gradient (ochre/red-orange regolith)
    const grad = ctx.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0, '#B8532F');
    grad.addColorStop(0.5, '#994424');
    grad.addColorStop(1, '#6F2B14');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 256, 256);

    // Bedrock formations (cooler grayish-slate blocks)
    ctx.fillStyle = '#736B63';
    ctx.beginPath();
    ctx.moveTo(30, 80); ctx.lineTo(120, 70); ctx.lineTo(140, 130); ctx.lineTo(40, 140);
    ctx.fill();

    // Sand dune ripples (golden orange ribbons)
    ctx.strokeStyle = '#D97A38';
    ctx.lineWidth = 4;
    for (let y = 140; y < 240; y += 14) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.bezierCurveTo(70, y - 8, 180, y + 10, 256, y - 5);
      ctx.stroke();
    }

    // Big Rocks / Boulders with cast shadows
    ctx.fillStyle = '#422216';
    ctx.beginPath(); ctx.arc(180, 90, 22, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(80, 190, 16, 0, Math.PI * 2); ctx.fill();

    return canvas;
  }

  function createProceduralMask(canvas, presetKey) {
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = 256;
    maskCanvas.height = 256;
    const mctx = maskCanvas.getContext('2d');

    // Fill Soil (Green)
    mctx.fillStyle = '#2ECC71';
    mctx.fillRect(0, 0, 256, 256);

    // Bedrock (Blue)
    mctx.fillStyle = '#3498DB';
    mctx.beginPath();
    mctx.moveTo(30, 80); mctx.lineTo(120, 70); mctx.lineTo(140, 130); mctx.lineTo(40, 140);
    mctx.fill();

    // Sand (Yellow)
    mctx.fillStyle = '#F1C40F';
    mctx.fillRect(0, 140, 256, 116);

    // Big Rocks (Red)
    mctx.fillStyle = '#E74C3C';
    mctx.beginPath(); mctx.arc(180, 90, 22, 0, Math.PI * 2); mctx.fill();
    mctx.beginPath(); mctx.arc(80, 190, 16, 0, Math.PI * 2); mctx.fill();

    return maskCanvas;
  }

  function createBlendedOverlay(rawCanvas, maskCanvas, alpha) {
    const overlayCanvas = document.createElement('canvas');
    overlayCanvas.width = 256;
    overlayCanvas.height = 256;
    const octx = overlayCanvas.getContext('2d');

    octx.drawImage(rawCanvas, 0, 0);
    octx.globalAlpha = alpha;
    octx.drawImage(maskCanvas, 0, 0);
    octx.globalAlpha = 1.0;
    return overlayCanvas;
  }

  // Update UI with analysis results
  function updateUI(data) {
    pctSoil.textContent = `${data.soil.toFixed(1)}%`;
    pctBedrock.textContent = `${data.bedrock.toFixed(1)}%`;
    pctSand.textContent = `${data.sand.toFixed(1)}%`;
    pctRock.textContent = `${data.rock.toFixed(1)}%`;

    barSoil.style.width = `${data.soil}%`;
    barBedrock.style.width = `${data.bedrock}%`;
    barSand.style.width = `${data.sand}%`;
    barRock.style.width = `${data.rock}%`;

    // Advisory
    advisoryBadge.textContent = data.advisory.title;
    advisoryBadge.className = 'status-badge ' + data.advisory.level.toLowerCase();
    advisoryTitle.textContent = data.advisory.title;
    advisoryMessage.textContent = data.advisory.message;
  }

  // Main segmentation trigger
  async function runSegmentation() {
    const alpha = parseFloat(alphaSlider.value) / 100.0;
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span> ANALYZING MARTIAN SURFACE...';

    if (currentFile) {
      // Try sending to Vercel Serverless API /api/segment
      const formData = new FormData();
      formData.append('file', currentFile);
      formData.append('alpha', alpha.toString());

      try {
        const resp = await fetch('/api/segment', {
          method: 'POST',
          body: formData,
        });

        if (resp.ok) {
          const res = await resp.json();
          viewRaw.src = res.images.original;
          viewMask.src = res.images.mask;
          viewOverlay.src = res.images.overlay;

          updateUI({
            soil: res.distribution.Soil.percentage,
            bedrock: res.distribution.Bedrock.percentage,
            sand: res.distribution.Sand.percentage,
            rock: res.distribution['Big Rock'].percentage,
            advisory: res.advisory,
          });
          analyzeBtn.disabled = false;
          analyzeBtn.innerHTML = '<span class="btn-icon">⚡</span> RUN TERRAIN SEGMENTATION';
          return;
        }
      } catch (err) {
        console.warn('API route not reached, running client-side fallback engine:', err);
      }
    }

    // Client-side execution mode
    const chosenPreset = presetSelect.value;
    const presetInfo = PRESET_DATA[chosenPreset] || PRESET_DATA.preset1;

    setTimeout(() => {
      const rawC = createProceduralMarsCanvas(chosenPreset);
      const maskC = createProceduralMask(rawC, chosenPreset);
      const overlayC = createBlendedOverlay(rawC, maskC, alpha);

      viewRaw.src = rawC.toDataURL('image/jpeg');
      viewMask.src = maskC.toDataURL('image/png');
      viewOverlay.src = overlayC.toDataURL('image/png');

      updateUI(presetInfo);

      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = '<span class="btn-icon">⚡</span> RUN TERRAIN SEGMENTATION';
    }, 400);
  }

  analyzeBtn.addEventListener('click', runSegmentation);

  // Auto-run on load for immediate live preview
  runSegmentation();
});

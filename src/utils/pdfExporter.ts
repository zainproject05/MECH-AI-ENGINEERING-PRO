import html2canvas from "html2canvas";
import { ModelPerformance, PreprocessedData } from "./mlEngine";
import { ParsedDataset } from "./fileParser";

interface ExporterParams {
  dataset: ParsedDataset | null;
  selectedFeatures: string[];
  selectedTarget: string;
  preprocessingResult: PreprocessedData | null;
  performances: ModelPerformance[];
  bestModelName: string;
  aiAnalysis?: string;
  language?: "en" | "id";
}

export async function exportPremiumPDF({
  dataset,
  selectedFeatures,
  selectedTarget,
  preprocessingResult,
  performances,
  bestModelName,
  aiAnalysis = "",
  language = "id",
}: ExporterParams): Promise<void> {
  const tLocal = (en: string, id: string) => (language === "id" ? id : en);

  // Show a global toast or status indicator in the console
  console.log("Starting premium high-fidelity PDF compilation...");

  // 1. Gather charts from the DOM to include as sharp high-res pictures
  const chartImages: { [key: string]: string } = {};
  const chartIds = [
    "chart_line_sequence",
    "chart_model_comparison_bar",
    "chart_error_histogram",
    "chart_feature_importances",
    "chart_target_density"
  ];

  for (const id of chartIds) {
    const el = document.getElementById(id);
    if (el) {
      try {
        const canvas = await html2canvas(el, {
          backgroundColor: "#080911",
          scale: 2, // Double rendering resolution for ultra-sharp prints
          logging: false,
          useCORS: true,
          scrollX: 0,
          scrollY: 0,
          windowWidth: el.scrollWidth,
          windowHeight: el.scrollHeight,
        });
        chartImages[id] = canvas.toDataURL("image/png");
      } catch (err) {
        console.warn(`Could not capture chart with ID ${id} for PDF:`, err);
      }
    }
  }

  // 2. Build of a beautifully styled print container
  const printWindow = window.open("", "_blank");
  if (!printWindow) {
    alert(tLocal(
      "Pop-up blocked! Please allow popups to export the premium PDF report.",
      "Pop-up terblokir! Harap izinkan pop-up untuk mengekspor laporan PDF premium."
    ));
    return;
  }

  const bestModel = performances.find((p) => p.modelName === bestModelName) || performances[0];
  const numRows = preprocessingResult?.allX.length || dataset?.rows.length || 0;
  const numFeatures = selectedFeatures.length;

  const datasetName = dataset?.fileName || tLocal("Mechanical Custom Dataset", "Kumpulan Data Teknik Mekanikal");
  const studentName = "ANANDA NUR DAFFA ZAIN";
  const studentNim = "20230130023";
  const university = "Universitas Muhammadiyah Yogyakarta (UMY)";
  const faculty = tLocal("Department of Mechanical Engineering", "Teknik Mesin · Universitas Muhammadiyah Yogyakarta");

  // Prepare Comparative Table content safely
  const tableRowsHtml = performances
    .map((perf, index) => {
      const isBest = perf.modelName === bestModelName;
      return `
      <tr style="${isBest ? "background-color: rgba(99, 102, 241, 0.08); font-weight: bold; border-left: 4px solid #6366f1;" : ""}">
        <td style="padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); font-family: sans-serif; text-align: left;">
          <div style="display: flex; items-center: center; gap: 8px;">
            <span style="width: 8px; height: 8px; border-radius: 50%; display: inline-block; background-color: ${isBest ? "#10b981" : "#475569"}"></span>
            <span style="color: #ffffff;">${perf.modelName}</span>
          </div>
        </td>
        <td style="padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); color: ${isBest ? "#10b981" : "#e2e8f0"}; text-align: right; font-weight: bold;">${perf.r2.toFixed(4)}</td>
        <td style="padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); color: #cbd5e1; text-align: right;">${perf.mae.toFixed(4)}</td>
        <td style="padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); color: #cbd5e1; text-align: right;">${perf.rmse.toFixed(4)}</td>
        <td style="padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); color: #cbd5e1; text-align: right;">${(perf.mape * 100).toFixed(2)}%</td>
        <td style="padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); color: #38bdf8; text-align: right;">${perf.trainingTime.toFixed(1)} ms</td>
        <td style="padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); color: #a5b4fc; text-align: right;">${perf.predictionTime.toFixed(2)} ms</td>
        <td style="padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: center;">
          <span style="background-color: ${isBest ? "rgba(16,185,129,0.15)" : "rgba(255,255,255,0.05)"}; color: ${isBest ? "#34d399" : "#94a3b8"}; border: 1px solid ${isBest ? "rgba(16,185,129,0.3)" : "rgba(255,255,255,0.1)"}; padding: 3px 8px; border-radius: 9999px; font-size: 9px; text-transform: uppercase; font-family: monospace;">
            ${isBest ? tLocal("CHAMPION", "JUARA") : perf.status}
          </span>
        </td>
      </tr>
    `;
    })
    .join("");

  // Prepare AI analysis content safely
  const formattedAiAnalysis = aiAnalysis
    ? aiAnalysis
        .split("\n")
        .map((paragraph) => {
          const trimmed = paragraph.trim();
          if (trimmed === "") return "";
          if (trimmed.startsWith("###")) {
            return `<h4 style="color:#ffffff; font-family: monospace; font-size:12px; text-transform: uppercase; letter-spacing:1px; margin-top:20px; margin-bottom:8px; border-left:3px solid #f59e0b; padding-left:8px;">${trimmed.replace(/^###\s*/, "")}</h4>`;
          }
          if (trimmed.startsWith("##")) {
            return `<h3 style="color:#ffffff; font-family: monospace; font-size:13px; text-transform: uppercase; letter-spacing:1.5px; margin-top:22px; margin-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:4px;">${trimmed.replace(/^##\s*/, "")}</h3>`;
          }
          if (trimmed.startsWith("-") || trimmed.startsWith("*")) {
            return `<li style="color:#94a3b8; font-size:12px; line-height:1.6; margin-left:16px; margin-bottom:6px; list-style-type: square;">${trimmed.replace(/^[-*]\s*/, "")}</li>`;
          }
          return `<p style="color:#94a3b8; font-size:12px; line-height:1.6; margin-bottom:12px; text-align: justify;">${trimmed}</p>`;
        })
        .join("")
    : `<p style="color:#64748b; font-size:12px; text-align: center; padding: 20px;">${tLocal(
        "No academic AI advisor analysis was generated at export compilation time.",
        "Analisis penasihat AI akademik tidak tersedia pada waktu pengesetan kompilasi laporan."
      )}</p>`;

  // Exporters document stylesheet & DOM structure
  printWindow.document.write(`
    <!DOCTYPE html>
    <html lang="id">
    <head>
      <meta charset="UTF-8">
      <title>MECHAUTOML SYSTEM DIAGNOSTICS REPORT</title>
      <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
        
        * {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }

        body {
          background-color: #05060b;
          color: #f8fafc;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          font-size: 13px;
          line-height: 1.5;
          padding: 30px;
          -webkit-print-color-adjust: exact;
          print-color-adjust: exact;
        }

        /* Continuous Premium styling for Report Cards */
        .page {
          max-width: 1024px;
          margin: 0 auto;
          background-color: #080a14;
          border: 1px solid rgba(99, 102, 241, 0.15);
          border-radius: 24px;
          padding: 40px;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
          position: relative;
          overflow: hidden;
          margin-bottom: 40px;
        }

        .page-break {
          page-break-after: always;
        }

        /* Gold & Neon Blue accent borders */
        .page::before {
          content: "";
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 4px;
          background: linear-gradient(90deg, #c2410c, #f59e0b, #6366f1, #10b981);
        }

        /* Header block */
        .report-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          border-bottom: 1px solid rgba(255,255,255,0.08);
          padding-bottom: 24px;
          margin-bottom: 30px;
        }

        .logo-box {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .logo-badge {
          background: linear-gradient(135deg, #1e1b4b 0%, #030712 100%);
          border: 2px solid #6366f1;
          color: #818cf8;
          font-family: 'JetBrains Mono', monospace;
          font-weight: 900;
          font-size: 18px;
          width: 48px;
          height: 48px;
          border-radius: 14px;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0px 4px 10px rgba(99,102,241,0.3);
        }

        .title-text h1 {
          font-size: 18px;
          font-weight: 900;
          text-transform: uppercase;
          letter-spacing: 1px;
          background: linear-gradient(90deg, #ffffff, #a5b4fc);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .title-text p {
          color: #94a3b8;
          font-size: 11px;
          font-weight: 500;
          margin-top: 4px;
        }

        .meta-card {
          background: linear-gradient(135deg, #0d0f1a 0%, #05060b 100%);
          border: 1px solid rgba(255,255,255,0.05);
          border-radius: 16px;
          padding: 16px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10.5px;
          color: #94a3b8;
          line-height: 1.6;
          text-align: right;
          box-shadow: inset 2px 2px 5px rgba(0,0,0,0.4);
        }

        .meta-card .accent-text {
          color: #f59e0b;
          font-weight: bold;
        }

        .section-title {
          font-size: 13px;
          font-weight: 900;
          font-family: 'JetBrains Mono', monospace;
          text-transform: uppercase;
          letter-spacing: 2px;
          color: #ffffff;
          margin-bottom: 20px;
          display: flex;
          align-items: center;
          gap: 8px;
          border-left: 4px solid #6366f1;
          padding-left: 10px;
        }

        /* 3D neumorphic scores */
        .widget-grid {
          display: grid;
          grid-template-cols: repeat(4, 1fr);
          gap: 15px;
          margin-bottom: 30px;
        }

        .widget-card {
          background: linear-gradient(135deg, #10121e 0%, #06070c 100%);
          border: 1px solid rgba(255,255,255,0.04);
          border-radius: 16px;
          padding: 16px;
          text-align: center;
          box-shadow: 10px 10px 25px rgba(0,0,0,0.45), inset 1.5px 1.5px 3px rgba(255,255,255,0.03);
          position: relative;
        }

        .widget-card span {
          display: block;
          font-family: 'JetBrains Mono', monospace;
          font-size: 8px;
          text-transform: uppercase;
          letter-spacing: 1px;
          color: #94a3b8;
          margin-bottom: 8px;
        }

        .widget-card .value {
          font-family: 'JetBrains Mono', monospace;
          font-size: 20px;
          font-weight: 900;
          color: #ffffff;
          letter-spacing: -0.5px;
        }

        .widget-card .desc {
          margin-top: 6px;
          font-size: 9px;
          color: #64748b;
          font-weight: 500;
        }

        /* Comparative matrix layout */
        .table-container {
          background-color: rgba(2,3,5,0.6);
          border: 1px solid rgba(255,255,255,0.06);
          border-radius: 16px;
          overflow: hidden;
          box-shadow: inset 4px 4px 12px rgba(0,0,0,0.5);
          margin-bottom: 30px;
        }

        table {
          width: 100%;
          border-collapse: collapse;
          font-size: 11px;
        }

        th {
          background-color: #0c0e17;
          color: #94a3b8;
          font-family: 'JetBrains Mono', monospace;
          font-size: 9px;
          font-weight: 900;
          text-transform: uppercase;
          letter-spacing: 1px;
          padding: 14px 16px;
          border-bottom: 1px solid rgba(255,255,255,0.08);
        }

        /* AI brain section layout */
        .advisor-panel {
          background: linear-gradient(135deg, #090b14 0%, #030407 100%);
          border: 1px solid rgba(255,255,255,0.05);
          border-radius: 20px;
          padding: 24px;
          box-shadow: 12px 12px 30px rgba(0,0,0,0.45);
          margin-bottom: 30px;
        }

        /* Chart placements to ensure they never clip and reside perfectly in columns */
        .charts-container {
          display: grid;
          grid-template-cols: 1fr 1fr;
          gap: 20px;
          margin-bottom: 24px;
        }

        .chart-box {
          background: #06070c;
          border: 1px solid rgba(255,255,255,0.04);
          border-radius: 16px;
          padding: 16px;
          text-align: center;
          height: 250px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          box-shadow: inset 2px 2px 6px rgba(0,0,0,0.6);
          overflow: hidden;
        }

        .chart-box h5 {
          font-family: 'JetBrains Mono', monospace;
          font-size: 9.5px;
          text-transform: uppercase;
          color: #cbd5e1;
          margin-bottom: 12px;
          letter-spacing: 1px;
          width: 100%;
          text-align: left;
          padding-left: 4px;
          border-left: 2px solid #10b981;
        }

        .chart-img {
          max-width: 100%;
          max-height: 190px;
          object-fit: contain;
          border-radius: 8px;
        }

        .no-chart-placeholder {
          color: #475569;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
        }

        .btn-print {
          position: fixed;
          bottom: 30px;
          right: 30px;
          background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
          border: none;
          color: white;
          padding: 12px 24px;
          border-radius: 9999px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          font-weight: 900;
          text-transform: uppercase;
          letter-spacing: 1px;
          cursor: pointer;
          box-shadow: 0 10px 20px rgba(99,102,241,0.4);
          display: flex;
          align-items: center;
          gap: 8px;
          transition: transform 0.2s;
          z-index: 1000;
        }

        .btn-print:hover {
          transform: translateY(-2px);
        }

        /* Print media breaks and overrides */
        @media print {
          body {
            background-color: #ffffff !important;
            color: #000000 !important;
            padding: 0 !important;
          }

          .page {
            max-width: 100% !important;
            border: none !important;
            background-color: #ffffff !important;
            box-shadow: none !important;
            padding: 20px !important;
            margin-bottom: 0 !important;
          }

          .page::before {
            display: none;
          }

          .report-header {
            border-bottom: 2px solid #000000 !important;
          }

          .title-text h1 {
            background: none !important;
            -webkit-text-fill-color: initial !important;
            color: #000000 !important;
          }

          .meta-card {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            color: #475569 !important;
            box-shadow: none !important;
          }

          .section-title {
            color: #000000 !important;
            border-left-color: #000000 !important;
          }

          .widget-card {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: none !important;
          }

          .widget-card span {
            color: #64748b !important;
          }

          .widget-card .value {
            color: #000000 !important;
          }

          .table-container {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: none !important;
          }

          th {
            background-color: #f1f5f9 !important;
            color: #000000 !important;
            border-bottom-color: #cbd5e1 !important;
          }

          td {
            color: #000000 !important;
            border-bottom-color: #cbd5e1 !important;
          }

          tr {
            background: none !important;
          }

          .advisor-panel {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: none !important;
          }

          .advisor-panel * {
            color: #000000 !important;
          }

          .btn-print {
            display: none !important;
          }

          .chart-box {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: none !important;
          }

          .chart-box h5 {
            color: #000000 !important;
            border-left-color: #000000 !important;
          }

          .chart-img {
            filter: invert(1) hue-rotate(180deg) brightness(1.2); /* Invert dark-mode charts for paper friendliness */
          }
        }
      </style>
    </head>
    <body>

      <!-- PAGE 1: EXECUTIVE BRIEF & PERFORMANCE STATISTICS -->
      <div class="page page-break">
        <div class="report-header">
          <div class="logo-box">
            <div class="logo-badge">M</div>
            <div class="title-text">
              <h1>MechAutoML</h1>
              <p>${tLocal("HIGH-FIDELITY DIAGNOSTICS & MACHINE EVALUATIONS", "LAPORAN DIAGNOSTIK KINERJA & EVALUASI MESIN PREMIUM")}</p>
            </div>
          </div>
          <div class="meta-card">
            <div>${tLocal("STUDENT", "MAHASISWA")}: <span class="accent-text">${studentName}</span></div>
            <div>NIM: <span>${studentNim}</span></div>
            <div>${tLocal("PROJECT", "PENELITIAN")}: <span>Applied AI in Manufacturing Systems</span></div>
            <div>${tLocal("FACULTY", "FAKULTAS")}: <span style="font-size: 8px;">${faculty}</span></div>
          </div>
        </div>

        <div class="section-title">
          <span>01 · ${tLocal("Executive Project Scope", "Cakupan Proyek & Detail Metadata")}</span>
        </div>
        
        <div class="widget-grid" style="grid-template-cols: repeat(4, 1fr); margin-bottom: 24px;">
          <div style="background: linear-gradient(135deg, #0f172a 0%, #020617 100%); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 12px; text-align: left; font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #94a3b8; box-shadow: 4px 4px 10px rgba(0,0,0,0.2);">
            <div style="font-size: 8px; text-transform: uppercase; color: #64748b; margin-bottom: 4px;">Workbook File</div>
            <div style="font-weight: bold; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${datasetName}">${datasetName}</div>
          </div>
          <div style="background: linear-gradient(135deg, #0f172a 0%, #020617 100%); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 12px; text-align: left; font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #94a3b8; box-shadow: 4px 4px 10px rgba(0,0,0,0.2);">
            <div style="font-size: 8px; text-transform: uppercase; color: #64748b; margin-bottom: 4px;">Ingested Records</div>
            <div style="font-weight: bold; color: #ffffff;">${numRows} Rows</div>
          </div>
          <div style="background: linear-gradient(135deg, #0f172a 0%, #020617 100%); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 12px; text-align: left; font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #94a3b8; box-shadow: 4px 4px 10px rgba(0,0,0,0.2);">
            <div style="font-size: 8px; text-transform: uppercase; color: #64748b; margin-bottom: 4px;">Predictor Features</div>
            <div style="font-weight: bold; color: #ffffff;">${numFeatures} Parameters</div>
          </div>
          <div style="background: linear-gradient(135deg, #0f172a 0%, #020617 100%); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 12px; text-align: left; font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #94a3b8; box-shadow: 4px 4px 10px rgba(0,0,0,0.2);">
            <div style="font-size: 8px; text-transform: uppercase; color: #64748b; margin-bottom: 4px;">Target Objective</div>
            <div style="font-weight: bold; color: #f59e0b;">${selectedTarget}</div>
          </div>
        </div>

        <div class="section-title">
          <span>02 · ${tLocal("Optimal Champion Estimator Performance", "Metrik Utama Model Terbaik (Champion Model)")}</span>
        </div>

        <div class="widget-grid">
          <div class="widget-card" style="border-top: 3px solid #10b981;">
            <span>R² Fit score</span>
            <div class="value" style="color: #34d399;">${bestModel.r2.toFixed(4)}</div>
            <div class="desc">${tLocal("Explains " + (bestModel.r2 * 100).toFixed(1) + "% of variance mapping", "Menjelaskan " + (bestModel.r2 * 100).toFixed(1) + "% pemetaan varians")}</div>
          </div>
          <div class="widget-card" style="border-top: 3px solid #3b82f6;">
            <span>Avg MAE error</span>
            <div class="value">${bestModel.mae.toFixed(4)}</div>
            <div class="desc">${tLocal("Mean Absolute residual deviation", "Deviasi galat absolut rata-rata")}</div>
          </div>
          <div class="widget-card" style="border-top: 3px solid #6366f1;">
            <span>RMS error (RMSE)</span>
            <div class="value">${bestModel.rmse.toFixed(4)}</div>
            <div class="desc">${tLocal("Root squared deviation standard", "Standar deviasi akar kuadrat")}</div>
          </div>
          <div class="widget-card" style="border-top: 3px solid #c084fc;">
            <span>MAP error ratio</span>
            <div class="value">${(bestModel.mape * 100).toFixed(2)}%</div>
            <div class="desc">${tLocal("Mean Absolute bounds percentage", "Persentase batas absolut rata-rata")}</div>
          </div>
        </div>

        <div class="section-title">
          <span>03 · ${tLocal("Estimator Comparative Evaluation Matrix", "Matriks Evaluasi Kinerja Multi-Model")}</span>
        </div>

        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th style="text-align: left;">${tLocal("Regressor Profile", "Profil Regressor")}</th>
                <th style="text-align: right;">R² Score</th>
                <th style="text-align: right;">MAE</th>
                <th style="text-align: right;">RMSE</th>
                <th style="text-align: right;">MAPE</th>
                <th style="text-align: right;">${tLocal("Train Time", "Waktu Latih")}</th>
                <th style="text-align: right;">${tLocal("Pred Time", "Waktu Pred")}</th>
                <th style="text-align: center;">Status</th>
              </tr>
            </thead>
            <tbody>
              ${tableRowsHtml}
            </tbody>
          </table>
        </div>

        <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #475569; margin-top: 15px;">
          PAGE 1 OF 3 · ${tLocal("COMPILED VIA MECHAUTOML KERNEL", "DIREKAYASA MELALUI COMPILER MECHAUTOML")}
        </div>
      </div>

      <!-- PAGE 2: HIGH-FIDELITY CHARTS & PLOTS -->
      <div class="page page-break">
        <div class="report-header">
          <div class="logo-box">
            <div class="logo-badge">V</div>
            <div class="title-text">
              <h1>Visual Analytics</h1>
              <p>${tLocal("GRAPHICAL PREDICTIVE MAPS & BIAS VERIFICATIONS", "DOKUMENTASI GRAFIS DIAGNOSTIK PREDIKTIF & PILA LENGKAP")}</p>
            </div>
          </div>
          <div class="meta-card">
            <div>${tLocal("FILE", "BERKAS")}: <span style="color: #ffffff;">${datasetName}</span></div>
            <div>${tLocal("ALGORITHM", "METODE")}: <span style="color: #10b981; font-weight: bold;">${bestModelName}</span></div>
          </div>
        </div>

        <div class="section-title">
          <span>04 · ${tLocal("Graphical Diagnostics (Validation Set Plots)", "Analisis Grafis & Verifikasi Penyebaran Data")}</span>
        </div>

        <!-- 2x2 Clean layout with strict bounds preventing clipping -->
        <div class="charts-container">
          <div class="chart-box">
            <h5>1. Verification Sequence (Actual vs Predicted)</h5>
            ${
              chartImages["chart_line_sequence"]
                ? `<img class="chart-img" src="${chartImages["chart_line_sequence"]}" alt="Validation Sequence Plot">`
                : `<div class="no-chart-placeholder">
                    <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
                    <span>Verification Sequence chart active on parent screen</span>
                   </div>`
            }
          </div>

          <div class="chart-box">
            <h5>2. Estimator R² Comparison Ranking</h5>
            ${
              chartImages["chart_model_comparison_bar"]
                ? `<img class="chart-img" src="${chartImages["chart_model_comparison_bar"]}" alt="Estimators Comparison">`
                : `<div class="no-chart-placeholder">
                    <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
                    <span>R² Comparison bar chart active on parent screen</span>
                   </div>`
            }
          </div>

          <div class="chart-box">
            <h5>3. Autocorrelation & Error Distribution Fit</h5>
            ${
              chartImages["chart_error_histogram"]
                ? `<img class="chart-img" src="${chartImages["chart_error_histogram"]}" alt="Error Histogram Gau Fit">`
                : `<div class="no-chart-placeholder">
                    <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                    <span>Residual error gaussian curve active on parent screen</span>
                   </div>`
            }
          </div>

          <div class="chart-box">
            <h5>4. Parametric Relative Influence Coefficients</h5>
            ${
              chartImages["chart_feature_importances"]
                ? `<img class="chart-img" src="${chartImages["chart_feature_importances"]}" alt="Feature Importances">`
                : `<div class="no-chart-placeholder">
                    <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"/></svg>
                    <span>Relative feature importances chart active on parent screen</span>
                   </div>`
            }
          </div>
        </div>

        <div style="background-color: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; font-family: monospace; font-size: 9.5px; color: #64748b; line-height: 1.5; text-align: justify;">
          <strong>${tLocal("DIAGNOSTIC NOTICE", "CATATAN INTERPRERASI GRAFIS")}:</strong> 
          ${tLocal(
            "The validation plots above evaluate the reliability of regression model projections. Chart 1 monitors sequential tracking capabilities, Chart 2 compares index limits, Chart 3 measures general gaussian fits, and Chart 4 maps coefficient influences on machining outcomes.",
            "Visualisasi analitis di atas merekam ketepatan prediksi model validasi. Gambar 1 memonitor presisi peramalan sampel sekuensial, Gambar 2 melacak perbandingan model, Gambar 3 memetakan bias galat teoretis kurva Bell, dan Gambar 4 menunjukkan pembobotan kontribusi parameter terhadap keluaran permesinan."
          )}
        </div>

        <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #475569; margin-top: 35px;">
          PAGE 2 OF 3 · ${tLocal("AUTOML DIGITAL TWIN DIAGNOSTIC", "SIMULASI DIGITAL TWIN DIAGNOSTIK KINERJA")}
        </div>
      </div>

      <!-- PAGE 3: INTUITION BIAS & ACADEMIC ANALYSIS -->
      <div class="page">
        <div class="report-header">
          <div class="logo-box">
            <div class="logo-badge">🧠</div>
            <div class="title-text">
              <h1>MECH AI Insights</h1>
              <p>${tLocal("ACADEMIC PHYSICAL REASONING & PHYSICAL DIAGNOSTICS", "LAPORAN INTERPRETASI SECARA FISIS DAN BIMBINGAN AKADEMIS")}</p>
            </div>
          </div>
          <div class="meta-card">
            <div>${tLocal("STUDENT", "MAHASISWA")}: <span style="color:#ffffff;">${studentName}</span></div>
            <div>NIM: <span>${studentNim}</span></div>
          </div>
        </div>

        <div class="section-title">
          <span>05 · ${tLocal("Executive Consultation Insights & Physical Correlations", "Bimbingan Akademis & Korelasi Parameter Fisika Permesinan")}</span>
        </div>

        <div class="advisor-panel">
          <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px;">
            <span style="font-size: 16px;">🎓</span>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: bold; color: #f59e0b;">
              MECH AI EXECUTIVE ADVISORY FEEDBACK · UMY
            </div>
          </div>
          <div style="max-height: 520px; overflow-y: visible; font-family: 'Inter', sans-serif;">
            ${formattedAiAnalysis}
          </div>
        </div>

        <div class="section-title">
          <span>06 · ${tLocal("Research Accountability & Honor Commitments", "Pernyataan Keaslian & Pernyataan Sikap Akademik")}</span>
        </div>
        
        <div style="font-style: italic; font-size: 11px; line-height: 1.6; color: #94a3b8; border-left: 3px solid #10b981; padding-left: 15px; margin-bottom: 15px;">
          &ldquo;As engineering students and practitioners, human oversight represents the final safeguard. AI tools provide powerful statistical calculations, but expert physical reasoning and peer-reviewed mechanical laws remain the absolute pillars of professional engineering excellence.&rdquo;
        </div>

        <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 30px; font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #475569;">
          <div>
            ${tLocal("Compiled on", "Disusun pada")}: ${new Date().toLocaleDateString("id-ID", {
              weekday: "long",
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </div>
          <div>
            PAGE 3 OF 3 · ${tLocal("END OF OFFICIAL METRICS REPORT", "AKHIR DARI DOKUMEN LAPORAN METRIK RESMI")}
          </div>
        </div>
      </div>

      <!-- PRINT ACTION TRACE FLOATER -->
      <button class="btn-print" onclick="window.print()">
        <svg width="16" height="16" fill="none" class="size-4" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
        </svg>
        <span>${tLocal("PRINT / SAVE AS PDF", "MAINKAN CETAK / SIMPAN PDF")}</span>
      </button>

      <script>
        // Trigger print system automatically
        window.addEventListener('load', () => {
          setTimeout(() => {
            window.print();
          }, 600);
        });
      </script>
    </body>
    </html>
  `);

  printWindow.document.close();
}

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        konjo: {
          bg: "#0a0c12",
          surface: "#11141c",
          "surface-2": "#181c27",
          line: "#232838",
          "line-soft": "#1a1e2a",
          fg: "#e7ecf4",
          "fg-muted": "#8a93a8",
          "fg-faint": "#4a5063",
          accent: "#7ad7ff",
          violet: "#b794ff",
          warm: "#f6c177",
          hot: "#ff4d6d",
          good: "#6ee7a3",
          cool: "#5fb3ff",
        },
      },
      fontFamily: {
        konjo: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        "konjo-display": ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        "konjo-mono": ["JetBrains Mono", "ui-monospace", "SF Mono", "Menlo", "monospace"],
      },
      borderRadius: {
        konjo: "10px",
        "konjo-lg": "16px",
      },
    },
  },
  plugins: [],
};

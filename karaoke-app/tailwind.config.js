/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {
    dropShadow: { glow: '0 0 8px rgba(253,224,71,0.7)' }
  }},
  plugins: [],
}
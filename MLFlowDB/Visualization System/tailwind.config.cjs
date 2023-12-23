/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{vue,js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      transitionProperty: {
        "border": "border-width",
      }
    },
  },
  important: true,
  plugins: [],
}

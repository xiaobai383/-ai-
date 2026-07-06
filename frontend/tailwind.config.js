/** @type {import('tailwindcss').Config} */
import tailwindcssAnimate from 'tailwindcss-animate'

export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#4F46E5',
          light: '#6366F1',
          lighter: '#818CF8',
        },
      },
      fontFamily: {
        sans: ['Noto Sans', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [tailwindcssAnimate],
}

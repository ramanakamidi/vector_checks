/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        cognizant: {
          blue: '#0033A1',
          'blue-dark': '#002878',
          'blue-light': '#1a5cc8',
          accent: '#00bfff',
          navy: '#1a1a2e',
        },
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};

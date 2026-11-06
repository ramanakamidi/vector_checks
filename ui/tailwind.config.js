/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        cognizant: {
          blue: '#3730A3',
          'blue-dark': '#211D66',
          'blue-light': '#4F46E5',
          accent: '#38BDF8',
          navy: '#151B3D',
        },
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};

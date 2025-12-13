#!/usr/bin/env python3
"""
THAU Developer Training - Entrenamiento especializado para generar código completo
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================================
# DATASET DE LANDING PAGES COMPLETAS
# ============================================================================

LANDING_PAGES_TRAINING = [
    {
        "prompt": "Crea una landing page completa para una startup de tecnología con sección hero, características, testimonios y formulario de contacto",
        "response": """Aquí está la landing page completa:

**index.html**
```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TechStartup - Innovación Digital</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <!-- Header -->
    <header class="header">
        <nav class="nav container">
            <a href="#" class="logo">TechStartup</a>
            <ul class="nav-links">
                <li><a href="#features">Características</a></li>
                <li><a href="#testimonials">Testimonios</a></li>
                <li><a href="#contact">Contacto</a></li>
            </ul>
        </nav>
    </header>

    <!-- Hero Section -->
    <section class="hero">
        <div class="container">
            <h1 class="hero-title">Transformamos ideas en realidad digital</h1>
            <p class="hero-subtitle">Soluciones tecnológicas innovadoras para tu negocio</p>
            <a href="#contact" class="btn btn-primary">Comenzar ahora</a>
        </div>
    </section>

    <!-- Features Section -->
    <section id="features" class="features">
        <div class="container">
            <h2 class="section-title">Nuestras Características</h2>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">🚀</div>
                    <h3>Velocidad</h3>
                    <p>Desarrollo rápido y eficiente</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔒</div>
                    <h3>Seguridad</h3>
                    <p>Protección de datos garantizada</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📱</div>
                    <h3>Responsive</h3>
                    <p>Funciona en todos los dispositivos</p>
                </div>
            </div>
        </div>
    </section>

    <!-- Testimonials Section -->
    <section id="testimonials" class="testimonials">
        <div class="container">
            <h2 class="section-title">Lo que dicen nuestros clientes</h2>
            <div class="testimonials-grid">
                <div class="testimonial-card">
                    <p class="testimonial-text">"Excelente servicio, superaron nuestras expectativas"</p>
                    <div class="testimonial-author">
                        <strong>María García</strong>
                        <span>CEO, Empresa ABC</span>
                    </div>
                </div>
                <div class="testimonial-card">
                    <p class="testimonial-text">"El mejor equipo de desarrollo con el que hemos trabajado"</p>
                    <div class="testimonial-author">
                        <strong>Juan López</strong>
                        <span>CTO, StartupXYZ</span>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Contact Section -->
    <section id="contact" class="contact">
        <div class="container">
            <h2 class="section-title">Contáctanos</h2>
            <form class="contact-form" id="contactForm">
                <div class="form-group">
                    <input type="text" name="name" placeholder="Tu nombre" required>
                </div>
                <div class="form-group">
                    <input type="email" name="email" placeholder="Tu email" required>
                </div>
                <div class="form-group">
                    <textarea name="message" placeholder="Tu mensaje" rows="4" required></textarea>
                </div>
                <button type="submit" class="btn btn-primary">Enviar mensaje</button>
            </form>
        </div>
    </section>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 TechStartup. Todos los derechos reservados.</p>
        </div>
    </footer>

    <script src="script.js"></script>
</body>
</html>
```

**styles.css**
```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --primary: #667eea;
    --secondary: #764ba2;
    --dark: #1a1a2e;
    --light: #f5f5f5;
    --white: #ffffff;
}

body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    line-height: 1.6;
    color: var(--dark);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Header */
.header {
    background: var(--white);
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    position: fixed;
    width: 100%;
    top: 0;
    z-index: 1000;
}

.nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 20px;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
    color: var(--primary);
    text-decoration: none;
}

.nav-links {
    display: flex;
    list-style: none;
    gap: 30px;
}

.nav-links a {
    text-decoration: none;
    color: var(--dark);
    transition: color 0.3s;
}

.nav-links a:hover {
    color: var(--primary);
}

/* Hero */
.hero {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: var(--white);
    padding: 150px 0 100px;
    text-align: center;
}

.hero-title {
    font-size: 3rem;
    margin-bottom: 20px;
    animation: fadeInUp 1s ease;
}

.hero-subtitle {
    font-size: 1.25rem;
    margin-bottom: 30px;
    opacity: 0.9;
    animation: fadeInUp 1s ease 0.2s backwards;
}

/* Buttons */
.btn {
    display: inline-block;
    padding: 12px 30px;
    border-radius: 30px;
    text-decoration: none;
    font-weight: bold;
    transition: transform 0.3s, box-shadow 0.3s;
    cursor: pointer;
    border: none;
}

.btn-primary {
    background: var(--white);
    color: var(--primary);
}

.btn-primary:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}

/* Features */
.features {
    padding: 80px 0;
    background: var(--light);
}

.section-title {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 50px;
    color: var(--dark);
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 30px;
}

.feature-card {
    background: var(--white);
    padding: 40px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    transition: transform 0.3s;
}

.feature-card:hover {
    transform: translateY(-10px);
}

.feature-icon {
    font-size: 3rem;
    margin-bottom: 20px;
}

.feature-card h3 {
    margin-bottom: 15px;
    color: var(--primary);
}

/* Testimonials */
.testimonials {
    padding: 80px 0;
}

.testimonials-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 30px;
}

.testimonial-card {
    background: var(--white);
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    border-left: 4px solid var(--primary);
}

.testimonial-text {
    font-style: italic;
    margin-bottom: 20px;
    font-size: 1.1rem;
}

.testimonial-author strong {
    display: block;
    color: var(--primary);
}

.testimonial-author span {
    font-size: 0.9rem;
    opacity: 0.7;
}

/* Contact */
.contact {
    padding: 80px 0;
    background: var(--light);
}

.contact-form {
    max-width: 600px;
    margin: 0 auto;
}

.form-group {
    margin-bottom: 20px;
}

.form-group input,
.form-group textarea {
    width: 100%;
    padding: 15px;
    border: 2px solid #ddd;
    border-radius: 10px;
    font-size: 1rem;
    transition: border-color 0.3s;
}

.form-group input:focus,
.form-group textarea:focus {
    outline: none;
    border-color: var(--primary);
}

/* Footer */
.footer {
    background: var(--dark);
    color: var(--white);
    padding: 30px 0;
    text-align: center;
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2rem;
    }

    .nav-links {
        display: none;
    }
}
```

**script.js**
```javascript
// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Form submission
document.getElementById('contactForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const data = Object.fromEntries(formData);

    // Show success message
    alert('¡Mensaje enviado correctamente! Te contactaremos pronto.');
    this.reset();
});

// Scroll animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

document.querySelectorAll('.feature-card, .testimonial-card').forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(30px)';
    card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(card);
});
```

Esta landing page incluye:
- Header fijo con navegación
- Hero section con gradiente y animaciones
- Sección de características con cards animadas
- Testimonios de clientes
- Formulario de contacto funcional
- Footer
- CSS responsive
- Animaciones al hacer scroll"""
    },
    {
        "prompt": "Crea un componente React de una landing page con Hero, Features y animaciones usando Tailwind CSS",
        "response": """Aquí está el componente React completo:

**src/components/LandingPage.jsx**
```jsx
import React, { useEffect, useRef } from 'react';

export default function LandingPage() {
  const featuresRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('animate-fade-in-up');
          }
        });
      },
      { threshold: 0.1 }
    );

    const cards = featuresRef.current?.querySelectorAll('.feature-card');
    cards?.forEach((card) => observer.observe(card));

    return () => observer.disconnect();
  }, []);

  const features = [
    { icon: '🚀', title: 'Velocidad', description: 'Rendimiento optimizado' },
    { icon: '🔒', title: 'Seguridad', description: 'Datos protegidos' },
    { icon: '📱', title: 'Responsive', description: 'Todos los dispositivos' },
    { icon: '⚡', title: 'Eficiencia', description: 'Ahorra tiempo y recursos' },
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-600 via-blue-600 to-cyan-500 opacity-90" />
        <div className="absolute inset-0">
          <div className="absolute top-20 left-20 w-72 h-72 bg-white/10 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-20 right-20 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse delay-1000" />
        </div>

        <div className="relative max-w-7xl mx-auto px-4 py-32 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6 animate-fade-in-up">
              <span className="block">Innovación</span>
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
                Digital
              </span>
            </h1>
            <p className="mt-6 max-w-2xl mx-auto text-xl text-gray-200 animate-fade-in-up delay-200">
              Transformamos ideas en soluciones tecnológicas que impulsan tu negocio hacia el futuro.
            </p>
            <div className="mt-10 flex justify-center gap-4 animate-fade-in-up delay-400">
              <button className="px-8 py-4 bg-white text-gray-900 font-bold rounded-full hover:scale-105 transition-transform duration-300 shadow-lg hover:shadow-xl">
                Comenzar
              </button>
              <button className="px-8 py-4 border-2 border-white/50 rounded-full hover:bg-white/10 transition-colors duration-300">
                Saber más
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section ref={featuresRef} className="py-24 bg-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">
              ¿Por qué elegirnos?
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Ofrecemos soluciones completas para tus necesidades digitales
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div
                key={index}
                className="feature-card opacity-0 p-8 bg-gray-900 rounded-2xl border border-gray-700 hover:border-purple-500 transition-all duration-300 hover:-translate-y-2 hover:shadow-lg hover:shadow-purple-500/20"
                style={{ transitionDelay: `${index * 100}ms` }}
              >
                <div className="text-5xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-bold mb-2 text-white">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-r from-purple-600 to-blue-600">
        <div className="max-w-4xl mx-auto text-center px-4">
          <h2 className="text-4xl font-bold mb-6">¿Listo para empezar?</h2>
          <p className="text-xl text-white/80 mb-8">
            Contáctanos hoy y llevemos tu proyecto al siguiente nivel
          </p>
          <button className="px-10 py-5 bg-white text-gray-900 font-bold text-lg rounded-full hover:scale-105 transition-transform duration-300 shadow-xl">
            Contáctanos →
          </button>
        </div>
      </section>
    </div>
  );
}
```

**tailwind.config.js**
```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      animation: {
        'fade-in-up': 'fadeInUp 0.6s ease-out forwards',
        'pulse': 'pulse 4s ease-in-out infinite',
      },
      keyframes: {
        fadeInUp: {
          '0%': {
            opacity: '0',
            transform: 'translateY(30px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
      },
    },
  },
  plugins: [],
};
```

Este componente incluye:
- Hero con gradientes y efectos de blur animados
- Animaciones al scroll usando IntersectionObserver
- Grid responsive de features
- Efectos hover en las cards
- CTA section con gradiente
- Totalmente responsive"""
    },
    {
        "prompt": "Genera un archivo HTML completo con una página de portfolio moderna con CSS incluido",
        "response": """Aquí está el portfolio completo:

**portfolio.html**
```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mi Portfolio</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --primary: #6366f1;
            --dark: #0f172a;
            --light: #f8fafc;
        }

        body {
            font-family: system-ui, sans-serif;
            background: var(--dark);
            color: var(--light);
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }

        /* Header */
        header {
            position: fixed;
            width: 100%;
            padding: 20px 0;
            background: rgba(15, 23, 42, 0.9);
            backdrop-filter: blur(10px);
            z-index: 100;
        }

        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--primary);
        }

        nav ul {
            display: flex;
            gap: 30px;
            list-style: none;
        }

        nav a {
            color: var(--light);
            text-decoration: none;
            transition: color 0.3s;
        }

        nav a:hover { color: var(--primary); }

        /* Hero */
        .hero {
            min-height: 100vh;
            display: flex;
            align-items: center;
            padding-top: 80px;
        }

        .hero-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 60px;
            align-items: center;
        }

        .hero h1 {
            font-size: 3.5rem;
            margin-bottom: 20px;
            line-height: 1.2;
        }

        .hero h1 span {
            color: var(--primary);
        }

        .hero p {
            font-size: 1.2rem;
            color: #94a3b8;
            margin-bottom: 30px;
        }

        .btn {
            display: inline-block;
            padding: 15px 35px;
            background: var(--primary);
            color: white;
            text-decoration: none;
            border-radius: 30px;
            font-weight: bold;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
        }

        .hero-image {
            position: relative;
        }

        .hero-image img {
            width: 100%;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }

        .hero-image::before {
            content: '';
            position: absolute;
            inset: -20px;
            border: 2px solid var(--primary);
            border-radius: 25px;
            z-index: -1;
            opacity: 0.5;
        }

        /* Projects */
        .projects {
            padding: 100px 0;
        }

        .section-title {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 60px;
        }

        .section-title span {
            color: var(--primary);
        }

        .projects-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
        }

        .project-card {
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            overflow: hidden;
            transition: transform 0.3s;
        }

        .project-card:hover {
            transform: translateY(-10px);
        }

        .project-image {
            height: 200px;
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
        }

        .project-info {
            padding: 25px;
        }

        .project-info h3 {
            font-size: 1.3rem;
            margin-bottom: 10px;
        }

        .project-info p {
            color: #94a3b8;
            margin-bottom: 15px;
        }

        .project-tags {
            display: flex;
            gap: 10px;
        }

        .tag {
            padding: 5px 12px;
            background: rgba(99, 102, 241, 0.2);
            color: var(--primary);
            border-radius: 15px;
            font-size: 0.85rem;
        }

        /* Skills */
        .skills {
            padding: 100px 0;
            background: rgba(255,255,255,0.02);
        }

        .skills-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 30px;
            text-align: center;
        }

        .skill-item {
            padding: 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            transition: transform 0.3s, background 0.3s;
        }

        .skill-item:hover {
            transform: scale(1.05);
            background: rgba(99, 102, 241, 0.1);
        }

        .skill-icon {
            font-size: 3rem;
            margin-bottom: 15px;
        }

        /* Contact */
        .contact {
            padding: 100px 0;
            text-align: center;
        }

        .contact p {
            color: #94a3b8;
            margin-bottom: 30px;
            font-size: 1.2rem;
        }

        /* Footer */
        footer {
            padding: 30px 0;
            text-align: center;
            border-top: 1px solid rgba(255,255,255,0.1);
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .hero-content > * {
            animation: fadeIn 1s ease forwards;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .hero-content {
                grid-template-columns: 1fr;
                text-align: center;
            }
            .hero h1 { font-size: 2.5rem; }
            nav ul { display: none; }
        }
    </style>
</head>
<body>
    <header>
        <nav class="container">
            <div class="logo">Portfolio</div>
            <ul>
                <li><a href="#projects">Proyectos</a></li>
                <li><a href="#skills">Habilidades</a></li>
                <li><a href="#contact">Contacto</a></li>
            </ul>
        </nav>
    </header>

    <section class="hero">
        <div class="container">
            <div class="hero-content">
                <div>
                    <h1>Hola, soy <span>Desarrollador</span> Full Stack</h1>
                    <p>Creo experiencias digitales modernas y funcionales que impulsan el éxito de tu negocio.</p>
                    <a href="#contact" class="btn">Contactar</a>
                </div>
                <div class="hero-image">
                    <div style="height:400px;background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:20px;display:flex;align-items:center;justify-content:center;font-size:5rem;">👨‍💻</div>
                </div>
            </div>
        </div>
    </section>

    <section id="projects" class="projects">
        <div class="container">
            <h2 class="section-title">Mis <span>Proyectos</span></h2>
            <div class="projects-grid">
                <div class="project-card">
                    <div class="project-image"></div>
                    <div class="project-info">
                        <h3>E-Commerce Platform</h3>
                        <p>Plataforma de comercio electrónico con carrito y pagos</p>
                        <div class="project-tags">
                            <span class="tag">React</span>
                            <span class="tag">Node.js</span>
                        </div>
                    </div>
                </div>
                <div class="project-card">
                    <div class="project-image" style="background:linear-gradient(135deg,#10b981,#059669);"></div>
                    <div class="project-info">
                        <h3>Dashboard Analytics</h3>
                        <p>Panel de control con visualización de datos en tiempo real</p>
                        <div class="project-tags">
                            <span class="tag">Vue.js</span>
                            <span class="tag">D3.js</span>
                        </div>
                    </div>
                </div>
                <div class="project-card">
                    <div class="project-image" style="background:linear-gradient(135deg,#f59e0b,#d97706);"></div>
                    <div class="project-info">
                        <h3>Mobile App</h3>
                        <p>Aplicación móvil multiplataforma para gestión de tareas</p>
                        <div class="project-tags">
                            <span class="tag">React Native</span>
                            <span class="tag">Firebase</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <section id="skills" class="skills">
        <div class="container">
            <h2 class="section-title">Mis <span>Habilidades</span></h2>
            <div class="skills-grid">
                <div class="skill-item">
                    <div class="skill-icon">⚛️</div>
                    <div>React</div>
                </div>
                <div class="skill-item">
                    <div class="skill-icon">🟢</div>
                    <div>Node.js</div>
                </div>
                <div class="skill-item">
                    <div class="skill-icon">🐍</div>
                    <div>Python</div>
                </div>
                <div class="skill-item">
                    <div class="skill-icon">🎨</div>
                    <div>CSS/Tailwind</div>
                </div>
                <div class="skill-item">
                    <div class="skill-icon">🗄️</div>
                    <div>PostgreSQL</div>
                </div>
                <div class="skill-item">
                    <div class="skill-icon">☁️</div>
                    <div>AWS</div>
                </div>
            </div>
        </div>
    </section>

    <section id="contact" class="contact">
        <div class="container">
            <h2 class="section-title">¿Trabajamos <span>juntos</span>?</h2>
            <p>Estoy disponible para proyectos freelance y oportunidades de trabajo</p>
            <a href="mailto:contacto@email.com" class="btn">Enviar email</a>
        </div>
    </section>

    <footer>
        <div class="container">
            <p>© 2024 Mi Portfolio. Hecho con ❤️</p>
        </div>
    </footer>
</body>
</html>
```

Este portfolio incluye:
- Diseño moderno oscuro
- Header fijo con blur
- Hero section con animaciones
- Grid de proyectos con hover effects
- Sección de skills
- CTA de contacto
- Totalmente responsive
- CSS incluido en el mismo archivo"""
    },
]

# ============================================================================
# DATASET DE COMPONENTES REACT
# ============================================================================

REACT_COMPONENTS_TRAINING = [
    {
        "prompt": "Crea un componente Card reutilizable con React y Tailwind",
        "response": """Aquí está el componente Card:

**src/components/Card.jsx**
```jsx
import React from 'react';

export function Card({
  children,
  className = '',
  hover = true,
  onClick,
  padding = 'p-6',
  rounded = 'rounded-xl',
  shadow = 'shadow-lg',
  border = false
}) {
  const baseStyles = `
    bg-white dark:bg-gray-800
    ${padding}
    ${rounded}
    ${shadow}
    ${border ? 'border border-gray-200 dark:border-gray-700' : ''}
    ${hover ? 'hover:shadow-xl hover:-translate-y-1 transition-all duration-300' : ''}
    ${onClick ? 'cursor-pointer' : ''}
    ${className}
  `;

  return (
    <div className={baseStyles} onClick={onClick}>
      {children}
    </div>
  );
}

export function CardHeader({ children, className = '' }) {
  return (
    <div className={`mb-4 ${className}`}>
      {children}
    </div>
  );
}

export function CardTitle({ children, className = '' }) {
  return (
    <h3 className={`text-xl font-bold text-gray-900 dark:text-white ${className}`}>
      {children}
    </h3>
  );
}

export function CardDescription({ children, className = '' }) {
  return (
    <p className={`text-gray-600 dark:text-gray-400 mt-2 ${className}`}>
      {children}
    </p>
  );
}

export function CardContent({ children, className = '' }) {
  return (
    <div className={className}>
      {children}
    </div>
  );
}

export function CardFooter({ children, className = '' }) {
  return (
    <div className={`mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 ${className}`}>
      {children}
    </div>
  );
}

// Uso:
// <Card>
//   <CardHeader>
//     <CardTitle>Mi Título</CardTitle>
//     <CardDescription>Descripción</CardDescription>
//   </CardHeader>
//   <CardContent>Contenido</CardContent>
//   <CardFooter>Footer</CardFooter>
// </Card>
```

Este componente incluye:
- Variantes configurables (hover, padding, shadow)
- Soporte para dark mode
- Subcomponentes (Header, Title, Content, Footer)
- Totalmente personalizable con className"""
    },
    {
        "prompt": "Crea un Modal/Dialog component con React",
        "response": """Aquí está el componente Modal:

**src/components/Modal.jsx**
```jsx
import React, { useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';

export function Modal({
  isOpen,
  onClose,
  children,
  title,
  size = 'md',
  closeOnOverlay = true,
  showCloseButton = true,
}) {
  const sizes = {
    sm: 'max-w-md',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
    full: 'max-w-full mx-4',
  };

  const handleEscape = useCallback((e) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, handleEscape]);

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Overlay */}
      <div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm transition-opacity"
        onClick={closeOnOverlay ? onClose : undefined}
      />

      {/* Modal Content */}
      <div
        className={`
          relative ${sizes[size]} w-full
          bg-white dark:bg-gray-800
          rounded-xl shadow-2xl
          transform transition-all duration-300
          animate-modal-enter
        `}
      >
        {/* Header */}
        {(title || showCloseButton) && (
          <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
            {title && (
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                {title}
              </h2>
            )}
            {showCloseButton && (
              <button
                onClick={onClose}
                className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        )}

        {/* Body */}
        <div className="p-6">
          {children}
        </div>
      </div>
    </div>,
    document.body
  );
}

export function ModalFooter({ children, className = '' }) {
  return (
    <div className={`flex justify-end gap-3 mt-6 pt-4 border-t border-gray-200 dark:border-gray-700 ${className}`}>
      {children}
    </div>
  );
}

// Agregar en globals.css:
// @keyframes modal-enter {
//   from { opacity: 0; transform: scale(0.95); }
//   to { opacity: 1; transform: scale(1); }
// }
// .animate-modal-enter { animation: modal-enter 0.2s ease-out; }

// Uso:
// const [isOpen, setIsOpen] = useState(false);
// <Modal isOpen={isOpen} onClose={() => setIsOpen(false)} title="Mi Modal">
//   <p>Contenido del modal</p>
//   <ModalFooter>
//     <button onClick={() => setIsOpen(false)}>Cancelar</button>
//     <button>Guardar</button>
//   </ModalFooter>
// </Modal>
```

Este modal incluye:
- Portal para renderizar fuera del DOM principal
- Cierre con Escape y click en overlay
- Múltiples tamaños
- Animación de entrada
- Bloqueo de scroll del body
- Soporte dark mode"""
    },
]

# ============================================================================
# MAIN
# ============================================================================

def create_training_dataset():
    """Crea el dataset combinado"""
    all_data = []

    for item in LANDING_PAGES_TRAINING:
        all_data.append({
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]}
            ]
        })

    for item in REACT_COMPONENTS_TRAINING:
        all_data.append({
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]}
            ]
        })

    return all_data


def save_dataset(data, output_path):
    """Guarda en formato JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Dataset guardado: {output_path} ({len(data)} ejemplos)")


def train_model(dataset_path):
    """Entrena con Ollama"""
    print("\n🚀 Creando thau:developer...")

    modelfile = """# THAU Developer - Especializado en generar código completo
FROM thau:advanced

SYSTEM \"\"\"Eres THAU Developer, un asistente de IA especializado en desarrollo web.

IMPORTANTE: Siempre generas código COMPLETO y FUNCIONAL, nunca fragmentos incompletos.

Para cada solicitud de desarrollo:
1. Genera archivos completos con todo el código
2. Usa el formato: **ruta/archivo.ext** seguido del bloque de código
3. Incluye HTML, CSS, JavaScript o React según corresponda
4. Añade animaciones y estilos modernos
5. Asegúrate de que el código sea funcional

Ejemplo de formato de respuesta:
**index.html**
```html
<!DOCTYPE html>
<html>...código completo...</html>
```

**styles.css**
```css
/* Estilos completos */
```

Nunca des explicaciones sin código. Siempre genera código funcional primero.
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
PARAMETER num_predict 4096
"""

    modelfile_path = Path(__file__).parent.parent / "Modelfile_thau_developer"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile)

    try:
        result = subprocess.run(
            ["ollama", "create", "thau:developer", "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print("✅ thau:developer creado")

            # Actualizar thau:latest
            subprocess.run(["ollama", "cp", "thau:developer", "thau:latest"], capture_output=True)
            print("✅ thau:latest actualizado")
        else:
            print(f"❌ Error: {result.stderr}")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    THAU Developer Training                                   ║
║           Entrenamiento especializado para código completo                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Crear dataset
    datasets_dir = Path(__file__).parent.parent / "data" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    data = create_training_dataset()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = datasets_dir / f"thau_developer_{timestamp}.jsonl"
    save_dataset(data, str(output_path))

    print(f"""
📊 Dataset creado:
   - Landing Pages: {len(LANDING_PAGES_TRAINING)} ejemplos
   - React Components: {len(REACT_COMPONENTS_TRAINING)} ejemplos
   - Total: {len(data)} ejemplos
""")

    train_model(str(output_path))
    print("\n✅ Entrenamiento completado!")


if __name__ == "__main__":
    main()

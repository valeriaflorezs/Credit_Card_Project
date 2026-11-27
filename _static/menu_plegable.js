// Menú lateral plegable: cada parte del libro («0. Introducción», «5. Modelo original»…) se contrae y se expande al
// pulsar su título. Arranca contraído salvo la parte de la página abierta; lo que el lector abre o cierra se recuerda.
(function () {
  'use strict';
  var KEY = 'cc-menu-partes';
  function leer() { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { return {}; } }
  function guardar(m) { try { localStorage.setItem(KEY, JSON.stringify(m)); } catch (e) { /* sin almacenamiento */ } }

  function iniciar() {
    // El primer enlace del menú es la portada y repetía el título del libro, que ya aparece arriba del menú.
    var inicio = document.querySelector('.bd-sidenav__home-link a');
    if (inicio) inicio.textContent = 'Portada';
    var nav = document.querySelector('.bd-docs-nav');
    if (!nav || nav.getAttribute('data-plegable')) return;
    nav.setAttribute('data-plegable', '1');
    var estado = leer(), partes = [];
    Array.prototype.forEach.call(nav.querySelectorAll('p.caption'), function (cap) {
      var lista = cap.nextElementSibling;
      if (!lista || lista.tagName !== 'UL') return;
      var nombre = (cap.textContent || '').trim();
      var activa = !!lista.querySelector('a.current, li.current, li.active, a[aria-current]');
      cap.classList.add('cc-parte');
      cap.setAttribute('role', 'button'); cap.setAttribute('tabindex', '0');
      function poner(abierta) {
        cap.classList.toggle('cc-cerrada', !abierta);
        cap.setAttribute('aria-expanded', abierta ? 'true' : 'false');
        lista.style.display = abierta ? '' : 'none';
      }
      function alternar() {
        var abre = cap.classList.contains('cc-cerrada');
        poner(abre); estado[nombre] = abre; guardar(estado);
      }
      cap.addEventListener('click', alternar);
      cap.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); alternar(); } });
      poner(activa || estado[nombre] === true);
      partes.push({ poner: poner, nombre: nombre });
    });
    if (!partes.length) return;
    var barra = document.createElement('div');
    barra.className = 'cc-menu-barra';
    [['Expandir todo', true], ['Contraer todo', false]].forEach(function (b) {
      var btn = document.createElement('button');
      btn.type = 'button'; btn.textContent = b[0];
      btn.addEventListener('click', function () {
        partes.forEach(function (p) { p.poner(b[1]); estado[p.nombre] = b[1]; });
        guardar(estado);
      });
      barra.appendChild(btn);
    });
    nav.insertBefore(barra, nav.firstChild);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', iniciar); else iniciar();
})();

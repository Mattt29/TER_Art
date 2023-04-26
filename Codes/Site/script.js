const toggler = document.querySelector(".hamburger");
const navLinksContainer = document.querySelector(".navlinks-container");

const toggleNav = e => {
  // Animation du bouton
  toggler.classList.toggle("open");

  const ariaToggle =
    toggler.getAttribute("aria-expanded") === "true" ? "false" : "true";
  toggler.setAttribute("aria-expanded", ariaToggle);

  // Slide de la navigation
  navLinksContainer.classList.toggle("open");
};

toggler.addEventListener("click", toggleNav);


new ResizeObserver(entries => {
  if (entries[0].contentRect.width <= 900){
    navLinksContainer.style.transition = "transform 0.4s ease-out";
  } else {
    navLinksContainer.style.transition = "none";
  }
}).observe(document.body)


const navLinks = document.querySelectorAll('.navlinks-container a');

navLinks.forEach((link) => {
  link.addEventListener('mouseenter', (e) => {
    const target = e.target;
    const letters = target.textContent.split('');
    target.textContent = '';
    let delay = 0;

    letters.forEach((letter, index) => {
      const span = document.createElement('span');
      span.textContent = letter;
      span.style.animationDelay = `${delay}ms`;
      target.appendChild(span);
      delay += 100;
    });
  });

  link.addEventListener('mouseleave', (e) => {
    const target = e.target;
    const spans = target.querySelectorAll('span');

    spans.forEach((span, index) => {
      setTimeout(() => {
        span.style.color = 'rgba(65, 65, 65, 1)';
      }, index * 100);
    });
  });
});

// INTRO

// Vérifier si l'animation a déjà été jouée
const animationPlayed = sessionStorage.getItem("animationPlayed");

// Si l'animation n'a pas été jouée, alors ajouter l'événement "DOMContentLoaded"
if (!animationPlayed || animationPlayed === "false") {
  document.addEventListener("DOMContentLoaded", async function () {
    const introAnimation = document.querySelector(".intro-animation");

    if (introAnimation && introAnimation.classList.contains("intro-animation-active")) {
      // Attendre que la vérification soit terminée
      await new Promise((resolve) => {
        sessionStorage.setItem("animationPlayed", "true");

        setTimeout(() => {
          introAnimation.style.opacity = "0";
          setTimeout(() => {
            introAnimation.style.display = "none";
            resolve(); // Résoudre la promesse pour indiquer que la vérification est terminée
          }, 4000);
        }, 3000);
      });
    }
  });
} else {
  const introAnimation = document.querySelector(".intro-animation");
  if (introAnimation) {
    introAnimation.style.display = "none";
  }
}

document.addEventListener("DOMContentLoaded", () => {

  // BASE.HTML

  const toggler = document.querySelector(".hamburger");
  const navLinksContainer = document.querySelector(".navlinks-container");

  if (toggler && navLinksContainer) {
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
      if (entries[0].contentRect.width <= 900) {
        navLinksContainer.style.transition = "transform 0.4s ease-out";
      } else {
        navLinksContainer.style.transition = "none";
      }
    }).observe(document.body)

  }

  // HOME.HTML

  const carousel = document.querySelector(".carousel");

  if (carousel) {
    const firstImg = carousel.querySelectorAll("img")[0];
    const arrowIcons = document.querySelectorAll(".wrapper i");

    let isDragStart = false, isDragging = false, prevPageX, prevScrollLeft, positionDiff;

    const showHideIcons = () => {
      // showing and hiding prev/next icon according to carousel scroll left value
      let scrollWidth = carousel.scrollWidth - carousel.clientWidth; // getting max scrollable width
      arrowIcons[0].style.display = carousel.scrollLeft == 0 ? "none" : "block";
      arrowIcons[1].style.display = carousel.scrollLeft == scrollWidth ? "none" : "block";
    }

    arrowIcons.forEach((icon) => {
      icon.addEventListener("click", () => {
        let firstImgWidth = firstImg.clientWidth + 14;
        const target = carousel.scrollLeft + (icon.id === "left" ? -firstImgWidth : firstImgWidth);
        smoothScroll(target, 1800);
        setTimeout(() => showHideIcons(), 60);
      });
    });

    const autoSlide = () => {
      // if there is no image left to scroll then return from here
      if (carousel.scrollLeft - (carousel.scrollWidth - carousel.clientWidth) > -1 || carousel.scrollLeft <= 0) return;


      positionDiff = Math.abs(positionDiff); // making positionDiff value to positive
      let firstImgWidth = firstImg.clientWidth + 14;
      // getting difference value that needs to add or reduce from carousel left to take middle img center
      let valDifference = firstImgWidth - positionDiff;

      if (carousel.scrollLeft > prevScrollLeft) { // if user is scrolling to the right
        return carousel.scrollLeft += positionDiff > firstImgWidth / 3 ? valDifference : -positionDiff;
      }
      // if user is scrolling to the left
      carousel.scrollLeft -= positionDiff > firstImgWidth / 3 ? valDifference : -positionDiff;
    }

    const dragStart = (e) => {
      // updatating global variables value on mouse down event
      isDragStart = true;
      prevPageX = e.pageX || e.touches[0].pageX;
      prevScrollLeft = carousel.scrollLeft;
    }

    const dragging = (e) => {
      // scrolling images/carousel to left according to mouse pointer
      if (!isDragStart) return;
      e.preventDefault();
      isDragging = true;
      carousel.classList.add("dragging");
      positionDiff = (e.pageX || e.touches[0].pageX) - prevPageX;
      carousel.scrollLeft = prevScrollLeft - positionDiff;
      showHideIcons();
    }

    const dragStop = () => {
      isDragStart = false;
      carousel.classList.remove("dragging");

      if (!isDragging) return;
      isDragging = false;
      autoSlide();
    }

    carousel.addEventListener("mousedown", dragStart);
    carousel.addEventListener("touchstart", dragStart);

    document.addEventListener("mousemove", dragging);
    carousel.addEventListener("touchmove", dragging);

    document.addEventListener("mouseup", dragStop);
    carousel.addEventListener("touchend", dragStop);

    const autoScroll = () => {
      const firstImgWidth = firstImg.clientWidth + 14;
      const target = carousel.scrollLeft + firstImgWidth;

      if (carousel.scrollLeft >= carousel.scrollWidth - carousel.clientWidth) {
        smoothScroll(0, 1800);
      } else {
        smoothScroll(target, 1800);
      }

      setTimeout(() => showHideIcons(), 60);
    };

    setInterval(autoScroll, 7000);

    document.addEventListener("mouseup", () => {
      dragStop();
      autoScroll();
    });
    carousel.addEventListener("touchend", () => {
      dragStop();
      autoScroll();
    });

    const smoothScroll = (target, duration) => {
      const start = carousel.scrollLeft;
      const change = target - start;
      let currentTime = 0;

      const animateScroll = () => {
        currentTime += 20;
        const val = easeInOutQuad(currentTime, start, change, duration);
        carousel.scrollLeft = val;

        if (currentTime < duration) {
          requestAnimationFrame(animateScroll);
        }
      };

      animateScroll();
    };

    const easeInOutQuad = (t, b, c, d) => {
      t /= d / 2;
      if (t < 1) return (c / 2) * t * t + b;
      t--;
      return (-c / 2) * (t * (t - 2) - 1) + b;
    };
  }
});
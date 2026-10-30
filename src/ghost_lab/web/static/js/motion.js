
/* Editorial motion for The Ghost Hunter's Lab.
   Animation enhances the recorded science; it never changes evidence.
*/

(() => {
  "use strict";

  const gsap = window.gsap;
  const ScrollTrigger = window.ScrollTrigger;

  // The page remains fully readable if GSAP is unavailable.
  if (!gsap || !ScrollTrigger) {
    return;
  }

  gsap.registerPlugin(ScrollTrigger);

  // GSAP automatically reverts these animations when motion preferences change.
  const media = gsap.matchMedia();

  media.add("(prefers-reduced-motion: no-preference)", () => {
    // 01 — Enter the investigation.
    const opening = gsap.timeline({
      defaults: {
        ease: "power3.out",
      },
    });

    opening
      .from(".masthead", {
        autoAlpha: 0,
        y: -18,
        duration: 0.8,
      })
      .from(
        ".hero .eyebrow",
        {
          autoAlpha: 0,
          y: 18,
          duration: 0.7,
        },
        "-=0.35",
      )
      .from(
        ".hero h1",
        {
          autoAlpha: 0,
          y: 34,
          duration: 1.1,
        },
        "-=0.35",
      )
      .from(
        ".hero .lead",
        {
          autoAlpha: 0,
          y: 20,
          duration: 0.9,
        },
        "-=0.55",
      )
      .from(
        ".hero .read-link, .hero-question",
        {
          autoAlpha: 0,
          y: 15,
          duration: 0.7,
          stagger: 0.15,
        },
        "-=0.45",
      );

    // 02 — Reveal the schematic model.
    const plate = document.querySelector(".plate");

    if (plate) {
      const model = gsap.timeline({
        defaults: { ease: "power2.out" },
        delay: 0.35,
      });

      model
        .from(".plate-heading", {
          autoAlpha: 0,
          y: 12,
          duration: 0.65,
        })
        .from(
          ".layer-ring",
          {
            autoAlpha: 0,
            duration: 0.55,
            stagger: {
              each: 0.025,
              from: "end",
            },
          },
          "-=0.2",
        )
        .from(
          ".target-ring",
          {
            autoAlpha: 0,
            scale: 0.92,
            transformOrigin: "50% 50%",
            duration: 1.2,
            ease: "power2.inOut",
          },
          "-=0.2",
        )
        .from(
          ".plate-leader, .plate-number, .plate-label",
          {
            autoAlpha: 0,
            y: 8,
            duration: 0.65,
            stagger: 0.12,
          },
          "-=0.45",
        )
        .from(
          ".plate figcaption",
          {
            autoAlpha: 0,
            y: 15,
            duration: 0.7,
          },
          "-=0.25",
        );

      // A gentle pulse suggests atmosphere, not measured activation.
      gsap.to(".target-ring", {
        opacity: 0.72,
        duration: 2.8,
        ease: "sine.inOut",
        repeat: -1,
        yoyo: true,
        delay: 2.4,
      });
    }

    // 03 — The evidence ledger appears after the opening.
    gsap.from(".evidence-strip p", {
      autoAlpha: 0,
      y: 20,
      duration: 0.8,
      stagger: 0.13,
      ease: "power2.out",
      scrollTrigger: {
        trigger: ".evidence-strip",
        start: "top 88%",
        once: true,
      },
    });

    // 04 — Each chapter unfolds as the reader enters it.
    document.querySelectorAll("[data-chapter]").forEach((chapter) => {
      const heading = chapter.querySelectorAll(
        ":scope > .chapter-index, :scope > h2",
      );

      if (heading.length) {
        gsap.from(heading, {
          autoAlpha: 0,
          y: 36,
          duration: 1,
          stagger: 0.14,
          ease: "power3.out",
          scrollTrigger: {
            trigger: chapter,
            start: "top 82%",
            once: true,
          },
        });
      }

      const content = chapter.querySelectorAll(
        ":scope > p:not(.chapter-index), :scope > blockquote",
      );

      if (content.length) {
        gsap.from(content, {
          autoAlpha: 0,
          y: 22,
          duration: 0.85,
          stagger: 0.12,
          ease: "power2.out",
          scrollTrigger: {
            trigger: chapter,
            start: "top 65%",
            once: true,
          },
        });
      }
    });

    // 05 — Find, Change, Compare.
    const steps = document.querySelector(".method-steps");

    if (steps) {
      gsap.from(".method-steps li", {
        autoAlpha: 0,
        y: 32,
        duration: 0.85,
        stagger: 0.2,
        ease: "power3.out",
        scrollTrigger: {
          trigger: steps,
          start: "top 82%",
          once: true,
        },
      });
    }

    // 06 — Reveal real measurements, not invented counters.
    const measurements = document.querySelector(".chapter--measurements");

    if (measurements) {
      gsap.from(".highlight-stat", {
        autoAlpha: 0,
        y: 25,
        duration: 0.95,
        ease: "power3.out",
        scrollTrigger: {
          trigger: ".highlight-stat",
          start: "top 85%",
          once: true,
        },
      });

      gsap.from(".measurement-table tbody tr", {
        autoAlpha: 0,
        duration: 0.55,
        stagger: 0.08,
        ease: "power1.out",
        scrollTrigger: {
          trigger: ".measurement-table",
          start: "top 82%",
          once: true,
        },
      });
    }

    // 07 — The unintended consequences.
    const consequences = document.querySelector(".side-effect-comparison");

    if (consequences) {
      gsap.from(".side-effect-comparison p", {
        autoAlpha: 0,
        y: 28,
        duration: 0.85,
        stagger: 0.2,
        ease: "power3.out",
        scrollTrigger: {
          trigger: consequences,
          start: "top 84%",
          once: true,
        },
      });
    }

    // 08 — The final scientific verdict.
    const verdict = document.querySelector(".chapter--verdict");

    if (verdict) {
      gsap.from(".chapter--verdict .read-link", {
        autoAlpha: 0,
        y: 16,
        duration: 0.85,
        ease: "power2.out",
        scrollTrigger: {
          trigger: verdict,
          start: "top 55%",
          once: true,
        },
      });
    }
  });
})();

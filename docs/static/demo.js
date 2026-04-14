(async function () {
  const config = window.PAPERPAGE_DEMO || {};
  const response = await fetch(config.examplesUrl, { cache: "no-store" });
  const payload = await response.json();
  const examples = payload.examples || [];
  const summary = payload.summary || {};

  const els = {
    index: document.getElementById("demo-index"),
    toolbarMeta: document.getElementById("demo-toolbar-meta"),
    title: document.getElementById("demo-title"),
    prompt: document.getElementById("demo-prompt"),
    promptPanels: document.getElementById("demo-prompt-panels"),
    media: document.getElementById("demo-media"),
    stageLayout: document.getElementById("demo-stage-layout"),
    compareAnswerSection: document.getElementById("demo-compare-answer-section"),
    compareAnswer: document.getElementById("demo-compare-answer"),
    random: document.getElementById("demo-random"),
    next: document.getElementById("demo-next"),
    summaryPills: document.getElementById("demo-summary-pills"),
    summaryBreakdown: document.getElementById("demo-summary-breakdown"),
    modeTabs: document.getElementById("demo-mode-tabs"),
    comparePanel: document.getElementById("demo-compare-panel"),
    comparePredictions: document.getElementById("demo-compare-predictions"),
    modelSelector: document.getElementById("demo-model-selector"),
    quizPanel: document.getElementById("demo-quiz-panel"),
    quizStats: document.getElementById("demo-quiz-stats"),
    form: document.getElementById("demo-form"),
    inputArea: document.getElementById("demo-input-area"),
    feedback: document.getElementById("demo-feedback"),
    answerSection: document.getElementById("demo-answer-section"),
    answer: document.getElementById("demo-answer"),
    quizModelSection: document.getElementById("demo-quiz-model-section"),
    quizModelHeading: document.getElementById("demo-quiz-model-heading"),
    quizModelCard: document.getElementById("demo-quiz-model-card"),
  };

  if (!examples.length) {
    els.title.textContent = "No examples available";
    els.prompt.textContent = "Add normalized examples to populate this demo.";
    return;
  }

  const availableModels =
    (config.availableModels && config.availableModels.length ? config.availableModels : payload.available_models) ||
    Array.from(
      new Set(
        examples.flatMap((example) => (example.predictions || []).map((prediction) => prediction.model).filter(Boolean))
      )
    );

  const enabledModes = (config.modes || ["compare", "quiz"]).filter(Boolean);
  const defaultMode = enabledModes.includes(config.defaultMode) ? config.defaultMode : enabledModes[0];
  const averagePromptLength =
    examples.reduce((total, example) => total + String(example.prompt || "").trim().length, 0) / examples.length;

  let currentMode = defaultMode;
  let currentIndex = 0;
  let activeCompareModels = new Set(
    (config.defaultCompareModels && config.defaultCompareModels.length
      ? config.defaultCompareModels
      : payload.default_compare_models) || availableModels.slice(0, 2)
  );
  if (!activeCompareModels.size && availableModels.length) {
    activeCompareModels = new Set([availableModels[0]]);
  }
  let selectedQuizModel =
    (config.quizRevealModel && availableModels.includes(config.quizRevealModel) && config.quizRevealModel) ||
    availableModels[0] ||
    "";
  const quizResults = new Map();
  let quizNavigationUnlocked = false;

  function normalizeText(value) {
    return String(value || "")
      .replace(/\\boxed\{([^}]*)\}/gi, "$1")
      .trim()
      .toLowerCase()
      .replace(/\s+/g, " ");
  }

  function canonicalVariants(value) {
    const base = normalizeText(value);
    const variants = new Set();
    if (!base) {
      return variants;
    }
    variants.add(base);
    variants.add(base.replace(/^[([{]\s*/, "").replace(/\s*[)\]}]$/, ""));
    variants.add(base.replace(/[()[\]{}]/g, ""));
    return variants;
  }

  function getAcceptedAnswers(example) {
    const answer = example.answer || {};
    const accepted = [answer.canonical].concat(answer.aliases || []).filter(Boolean);
    const variants = new Set();
    accepted.forEach((item) => {
      canonicalVariants(item).forEach((variant) => variants.add(variant));
    });
    return variants;
  }

  function getPredictionForModel(example, modelName) {
    return (example.predictions || []).find((prediction) => prediction.model === modelName) || null;
  }

  function getCompareGroups(example) {
    return Array.isArray(example.meta?.compare_groups) ? example.meta.compare_groups : [];
  }

  function hasGroupedCompare(example) {
    return getCompareGroups(example).length > 0;
  }

  function getVisibleCompareModels() {
    if (config.showCompareModelSelector === false) {
      return availableModels.slice();
    }
    return availableModels.filter((model) => activeCompareModels.has(model));
  }

  function renderSummaryPills() {
    els.summaryPills.innerHTML = "";
    const pills = [];
    if (summary.total_examples) {
      pills.push(`${summary.total_examples} examples`);
    }
    if (summary.tasks && typeof summary.tasks === "object") {
      pills.push(`${Object.keys(summary.tasks).length} tasks`);
    }
    if (Array.isArray(summary.prompt_families) && summary.prompt_families.length) {
      pills.push(`${summary.prompt_families.length} prompt families`);
    }
    pills.forEach((text) => {
      const pill = document.createElement("span");
      pill.className = "demo-summary-pill";
      pill.textContent = text;
      els.summaryPills.appendChild(pill);
    });
  }

  function renderSummaryBreakdown() {
    els.summaryBreakdown.innerHTML = "";
    if (config.showSummaryBreakdown === false) {
      els.summaryBreakdown.hidden = true;
      return;
    }
    const rows = Array.isArray(summary.accuracy_breakdown) ? summary.accuracy_breakdown : [];
    if (!rows.length) {
      els.summaryBreakdown.hidden = true;
      return;
    }

    const card = document.createElement("section");
    card.className = "demo-summary-breakdown-card";

    const title = document.createElement("h3");
    title.textContent = "Accuracy by Prompt Family and Rule";
    card.appendChild(title);

    const table = document.createElement("table");
    table.className = "demo-summary-breakdown-table";

    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    ["Prompt", "Rule"].concat(availableModels).forEach((label) => {
      const th = document.createElement("th");
      th.textContent = label;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    rows.forEach((row) => {
      const tr = document.createElement("tr");

      const promptCell = document.createElement("td");
      promptCell.textContent = row.prompt_family || "";
      tr.appendChild(promptCell);

      const ruleCell = document.createElement("td");
      ruleCell.textContent = row.rule_label || row.rule_variant || "";
      tr.appendChild(ruleCell);

      availableModels.forEach((model) => {
        const td = document.createElement("td");
        td.className = "is-metric";
        const stats = row.models?.[model];
        if (stats && typeof stats.accuracy === "number") {
          td.textContent = `${(stats.accuracy * 100).toFixed(1)}%`;
        } else {
          td.textContent = "—";
        }
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    card.appendChild(table);
    els.summaryBreakdown.appendChild(card);
    els.summaryBreakdown.hidden = false;
  }

  function renderModeTabs() {
    els.modeTabs.innerHTML = "";
    if (enabledModes.length <= 1) {
      els.modeTabs.hidden = true;
      return;
    }
    els.modeTabs.hidden = false;
    enabledModes.forEach((mode) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "demo-mode-tab" + (mode === currentMode ? " is-active" : "");
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", mode === currentMode ? "true" : "false");
      button.textContent = mode === "compare" ? "Compare" : mode === "quiz" ? "Quiz" : mode;
      button.addEventListener("click", () => {
        if (mode === currentMode) {
          return;
        }
        currentMode = mode;
        renderExample(currentIndex);
      });
      els.modeTabs.appendChild(button);
    });
  }

  function computeStageShare(example, img) {
    let mediaShare = 50;
    const promptLength = String(example.prompt || "").trim().length;

    if (averagePromptLength > 0) {
      if (promptLength < averagePromptLength * 0.75) {
        mediaShare += 8;
      } else if (promptLength < averagePromptLength * 0.9) {
        mediaShare += 4;
      } else if (promptLength > averagePromptLength * 1.35) {
        mediaShare -= 8;
      } else if (promptLength > averagePromptLength * 1.15) {
        mediaShare -= 4;
      }
    }

    if (img && img.naturalWidth && img.naturalHeight) {
      const aspectRatio = img.naturalWidth / img.naturalHeight;
      if (aspectRatio > 1.45) {
        mediaShare += 8;
      } else if (aspectRatio > 1.15) {
        mediaShare += 4;
      } else if (aspectRatio < 0.9) {
        mediaShare -= 6;
      }
    }

    mediaShare = Math.max(50, Math.min(66, mediaShare));
    return mediaShare;
  }

  function applyStageLayout(example, img) {
    const mediaShare = computeStageShare(example, img);
    els.stageLayout.style.setProperty("--demo-stage-media-width", `${mediaShare}%`);
  }

  function renderMedia(example) {
    els.media.innerHTML = "";
    const mediaItems = example.media || [];
    mediaItems.forEach((item) => {
      const card = document.createElement("figure");
      card.className = "demo-media-card";

      if (item.type === "image") {
        const img = document.createElement("img");
        img.src = item.src;
        img.alt = item.alt || example.title || example.task || "Demo example";
        img.addEventListener("load", () => applyStageLayout(example, img));
        card.appendChild(img);
      }

      if (item.caption) {
        const caption = document.createElement("figcaption");
        caption.className = "demo-media-caption";
        caption.textContent = item.caption;
        card.appendChild(caption);
      }
      els.media.appendChild(card);
    });

    const firstImg = els.media.querySelector("img");
    if (firstImg && firstImg.complete) {
      applyStageLayout(example, firstImg);
    } else {
      applyStageLayout(example, null);
    }
  }

  function renderInput(example) {
    els.inputArea.innerHTML = "";
    if (example.choices && example.choices.length) {
      const wrapper = document.createElement("div");
      wrapper.className = "demo-choice-list";
      example.choices.forEach((choice, index) => {
        const label = document.createElement("label");
        label.className = "demo-choice";

        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = "demo-choice";
        radio.value = choice.value || choice.text;
        radio.checked = index === 0;
        label.appendChild(radio);

        const textWrap = document.createElement("div");
        const choiceLabel = document.createElement("div");
        choiceLabel.className = "demo-choice-label";
        choiceLabel.textContent = choice.label || String.fromCharCode(65 + index);
        textWrap.appendChild(choiceLabel);

        const choiceText = document.createElement("div");
        choiceText.textContent = choice.text || choice.value;
        textWrap.appendChild(choiceText);
        label.appendChild(textWrap);

        wrapper.appendChild(label);
      });
      els.inputArea.appendChild(wrapper);
      return;
    }

    const input = document.createElement("input");
    input.className = "demo-text-input";
    input.type = "text";
    input.name = "demo-answer";
    input.placeholder = "Enter your answer";
    input.autocomplete = "off";
    els.inputArea.appendChild(input);
  }

  function renderPromptPanels(example) {
    els.promptPanels.innerHTML = "";
    if (hasGroupedCompare(example)) {
      els.promptPanels.hidden = true;
      return;
    }
    const panels = Array.isArray(example.meta?.prompt_panels) ? example.meta.prompt_panels : [];
    if (!panels.length) {
      els.promptPanels.hidden = true;
      return;
    }
    panels.forEach((panel) => {
      const wrap = document.createElement("section");
      wrap.className = "demo-prompt-panel";

      const heading = document.createElement("h4");
      heading.textContent = panel.label || "Prompt";
      wrap.appendChild(heading);

      const body = document.createElement("p");
      body.textContent = panel.text || "";
      wrap.appendChild(body);
      els.promptPanels.appendChild(wrap);
    });
    els.promptPanels.hidden = false;
  }

  function buildBadge(flag, textOverride) {
    const badge = document.createElement("span");
    const state =
      flag === null && textOverride && /correct/i.test(textOverride)
        ? "is-mixed"
        : flag === true
          ? "is-correct"
          : flag === false
            ? "is-incorrect"
            : "is-unknown";
    badge.className = "demo-badge " + state;
    badge.textContent =
      textOverride || (flag === true ? "Correct" : flag === false ? "Incorrect" : "Unscored");
    return badge;
  }

  function renderVariantList(container, variants, showBadges) {
    const list = document.createElement("div");
    list.className = "demo-variant-list";

    variants.forEach((variant) => {
      const row = document.createElement("div");
      row.className = "demo-variant-row";

      const head = document.createElement("div");
      head.className = "demo-variant-head";

      const label = document.createElement("div");
      label.className = "demo-variant-label";
      label.textContent = variant.label || "Variant";
      head.appendChild(label);

      if (showBadges) {
        head.appendChild(buildBadge(variant.correct));
      }
      row.appendChild(head);

      const answer = document.createElement("div");
      answer.className = "demo-variant-answer";
      answer.textContent = variant.answer || "";
      row.appendChild(answer);

      if (variant.explanation) {
        const expl = document.createElement("div");
        expl.className = "demo-variant-explanation";
        expl.textContent = variant.explanation;
        row.appendChild(expl);
      }
      list.appendChild(row);
    });

    container.appendChild(list);
  }

  function summarizePredictionVariants(variants) {
    const scored = variants.filter((variant) => variant.correct === true || variant.correct === false);
    if (!scored.length) {
      return { flag: null, text: "Unscored" };
    }
    const correct = scored.filter((variant) => variant.correct === true).length;
    if (correct === scored.length) {
      return { flag: true, text: `${correct}/${scored.length} correct` };
    }
    if (correct === 0) {
      return { flag: false, text: `0/${scored.length} correct` };
    }
    return { flag: null, text: `${correct}/${scored.length} correct` };
  }

  function renderAnswerCard(container, answer) {
    container.innerHTML = "";
    if (Array.isArray(answer.variants) && answer.variants.length) {
      renderVariantList(container, answer.variants, false);
    } else {
      const canonical = document.createElement("strong");
      canonical.textContent = answer.canonical || "No answer provided";
      container.appendChild(canonical);
    }
    if (answer.explanation) {
      const expl = document.createElement("div");
      expl.className = "demo-prediction-explanation";
      expl.textContent = answer.explanation;
      container.appendChild(expl);
    }
  }

  function renderPredictionCards(container, predictions) {
    container.innerHTML = "";
    predictions.forEach((prediction) => {
      const card = document.createElement("div");
      card.className = "demo-prediction-card";

      const top = document.createElement("div");
      top.className = "demo-prediction-top";
      const model = document.createElement("strong");
      model.textContent = prediction.model || "Model";
      top.appendChild(model);

      const hasVariants = Array.isArray(prediction.variants) && prediction.variants.length;
      const summaryBadge = hasVariants
        ? summarizePredictionVariants(prediction.variants)
        : {
            flag: prediction.correct === true ? true : prediction.correct === false ? false : null,
            text:
              prediction.correct === true
                ? "Correct"
                : prediction.correct === false
                  ? "Incorrect"
                  : "Unscored",
          };
      top.appendChild(buildBadge(summaryBadge.flag, summaryBadge.text));
      card.appendChild(top);

      if (hasVariants) {
        renderVariantList(card, prediction.variants, true);
      } else {
        const answer = document.createElement("div");
        answer.className = "demo-prediction-answer";
        answer.textContent = prediction.answer || "";
        card.appendChild(answer);
      }

      if (prediction.explanation) {
        const expl = document.createElement("div");
        expl.className = "demo-prediction-explanation";
        expl.textContent = prediction.explanation;
        card.appendChild(expl);
      }
      container.appendChild(card);
    });
  }

  function renderCompareSelector() {
    const example = examples[currentIndex];
    els.modelSelector.innerHTML = "";
    if (config.showCompareModelSelector === false || hasGroupedCompare(example)) {
      els.modelSelector.hidden = true;
      return;
    }
    els.modelSelector.hidden = false;
    availableModels.forEach((model) => {
      const label = document.createElement("label");
      label.className = "demo-model-pill" + (activeCompareModels.has(model) ? " is-active" : "");

      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = activeCompareModels.has(model);
      input.addEventListener("change", () => {
        if (input.checked) {
          activeCompareModels.add(model);
        } else if (activeCompareModels.size > 1) {
          activeCompareModels.delete(model);
        } else {
          input.checked = true;
        }
        renderCompareSelector();
        renderComparePanel();
      });
      label.appendChild(input);

      const text = document.createElement("span");
      text.textContent = model;
      label.appendChild(text);
      els.modelSelector.appendChild(label);
    });
  }

  function createPromptPanelNode(panel) {
    const wrap = document.createElement("section");
    wrap.className = "demo-prompt-panel";

    const heading = document.createElement("h4");
    heading.textContent = panel.label || "Prompt";
    wrap.appendChild(heading);

    const body = document.createElement("p");
    body.textContent = panel.text || "";
    wrap.appendChild(body);
    return wrap;
  }

  function renderGroupedComparePanel(example) {
    const visibleModels = getVisibleCompareModels();
    const groups = getCompareGroups(example);
    els.comparePredictions.innerHTML = "";
    els.comparePredictions.className = "demo-group-list";

    groups.forEach((group) => {
      const section = document.createElement("section");
      section.className = "demo-compare-group";

      const head = document.createElement("div");
      head.className = "demo-group-head";
      const titleWrap = document.createElement("div");
      const kicker = document.createElement("p");
      kicker.className = "demo-section-kicker";
      kicker.textContent = "Prompt Family";
      titleWrap.appendChild(kicker);
      const title = document.createElement("h4");
      title.textContent = group.label || "Prompt block";
      titleWrap.appendChild(title);
      head.appendChild(titleWrap);
      if (group.description) {
        const desc = document.createElement("p");
        desc.className = "demo-group-description";
        desc.textContent = group.description;
        head.appendChild(desc);
      }
      section.appendChild(head);

      const promptGrid = document.createElement("div");
      promptGrid.className = "demo-group-prompt-grid";
      (group.prompt_panels || []).forEach((panel) => {
        promptGrid.appendChild(createPromptPanelNode(panel));
      });
      section.appendChild(promptGrid);

      const predictionsWrap = document.createElement("div");
      predictionsWrap.className = "demo-predictions compare-grid compare-grid--models";
      const groupPredictions = visibleModels
        .map((model) => (group.predictions || []).find((prediction) => prediction.model === model))
        .filter(Boolean);
      renderPredictionCards(predictionsWrap, groupPredictions);
      section.appendChild(predictionsWrap);

      els.comparePredictions.appendChild(section);
    });
  }

  function renderComparePanel() {
    const example = examples[currentIndex];
    if (hasGroupedCompare(example)) {
      renderGroupedComparePanel(example);
      return;
    }
    const predictions = getVisibleCompareModels()
      .map((model) => getPredictionForModel(example, model))
      .filter(Boolean);
    els.comparePredictions.className = "demo-predictions compare-grid";
    renderPredictionCards(els.comparePredictions, predictions);
  }

  function renderQuizStats() {
    const attempted = quizResults.size;
    const correct = Array.from(quizResults.values()).filter(Boolean).length;
    const accuracy = attempted ? ((correct / attempted) * 100).toFixed(1) : "0.0";
    els.quizStats.innerHTML = "";

    const userPill = document.createElement("span");
    userPill.className = "demo-summary-pill";
    userPill.textContent = `You: ${correct}/${attempted} (${accuracy}%)`;
    els.quizStats.appendChild(userPill);
  }

  function renderQuizModelCard() {
    const example = examples[currentIndex];
    const prediction = getPredictionForModel(example, selectedQuizModel);
    if (!prediction) {
      els.quizModelSection.hidden = true;
      return;
    }
    els.quizModelHeading.textContent = `${selectedQuizModel} response`;
    renderPredictionCards(els.quizModelCard, [prediction]);
    els.quizModelSection.hidden = false;
  }

  function renderCompareAnswer() {
    const example = examples[currentIndex];
    const answer = example.answer || {};
    renderAnswerCard(els.compareAnswer, answer);
    els.compareAnswerSection.hidden = false;
  }

  function resetModePanelsForExampleChange() {
    els.comparePredictions.innerHTML = "";
    els.modelSelector.innerHTML = "";
    els.compareAnswerSection.hidden = true;
    els.compareAnswer.innerHTML = "";
    quizNavigationUnlocked = false;
    clearQuizFeedback();
  }

  function clearQuizFeedback() {
    els.feedback.hidden = true;
    els.feedback.className = "demo-feedback";
    els.answerSection.hidden = true;
    els.quizModelSection.hidden = true;
    els.quizModelCard.innerHTML = "";
  }

  function renderAnswer(example) {
    renderAnswerCard(els.answer, example.answer || {});
    els.answerSection.hidden = false;
  }

  function getSubmittedValue(example) {
    if (example.choices && example.choices.length) {
      const selected = els.form.querySelector('input[name="demo-choice"]:checked');
      return selected ? selected.value : "";
    }
    const input = els.form.querySelector("input[name='demo-answer']");
    return input ? input.value : "";
  }

  function scoreCurrentAnswer() {
    const example = examples[currentIndex];
    const accepted = getAcceptedAnswers(example);
    const submitted = canonicalVariants(getSubmittedValue(example));
    const hasSubmission = submitted.size > 0;
    const isCorrect =
      accepted.size && hasSubmission ? Array.from(submitted).some((item) => accepted.has(item)) : null;

    if (isCorrect !== null) {
      quizResults.set(example.id, isCorrect);
      renderQuizStats();
      els.feedback.hidden = false;
      if (isCorrect) {
        els.feedback.className = "demo-feedback is-correct";
        els.feedback.textContent = "Correct.";
      } else {
        els.feedback.className = "demo-feedback is-incorrect";
        els.feedback.textContent = "Not quite.";
      }
    } else {
      els.feedback.hidden = false;
      els.feedback.className = "demo-feedback";
      els.feedback.textContent = "Answer checking is unavailable for this example.";
    }

    renderAnswer(example);
    renderQuizModelCard();
    quizNavigationUnlocked = true;
    renderNavigationState();
  }

  function renderNavigationState() {
    const showNavigation = currentMode === "compare" || quizNavigationUnlocked;
    els.toolbarMeta.hidden = !showNavigation;
    els.random.disabled = !showNavigation;
    els.next.disabled = !showNavigation;
  }

  function renderPanels() {
    renderModeTabs();
    els.comparePanel.hidden = currentMode !== "compare";
    els.quizPanel.hidden = currentMode !== "quiz";
    if (currentMode === "compare") {
      renderCompareAnswer();
      renderCompareSelector();
      renderComparePanel();
    } else if (currentMode === "quiz") {
      els.compareAnswerSection.hidden = true;
      renderQuizStats();
    }
    renderNavigationState();
  }

  function nextRandomIndex() {
    if (examples.length <= 1) {
      return 0;
    }
    let nextIndex = Math.floor(Math.random() * examples.length);
    if (nextIndex === currentIndex) {
      nextIndex = (nextIndex + 1) % examples.length;
    }
    return nextIndex;
  }

  function renderExample(index) {
    currentIndex = ((index % examples.length) + examples.length) % examples.length;
    const example = examples[currentIndex];

    els.index.textContent = `Example ${currentIndex + 1} of ${examples.length}`;
    els.title.textContent = example.title || example.task || "Example";
    els.prompt.textContent = example.prompt || "";
    renderPromptPanels(example);
    resetModePanelsForExampleChange();
    renderMedia(example);
    renderInput(example);
    renderPanels();
  }

  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    scoreCurrentAnswer();
  });

  els.random.addEventListener("click", () => renderExample(nextRandomIndex()));
  els.next.addEventListener("click", () => renderExample(currentIndex + 1));

  renderSummaryPills();
  renderSummaryBreakdown();
  renderExample(0);
})();

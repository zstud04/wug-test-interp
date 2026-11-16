# EACL rebuttal: re-run the paper's cue-condition analysis (analysis/interp-analysis.R,
# fit2 and the best-layer novel-vs-real comparison) with the NEW vision-cued
# embeddings in place of the old ones. Language-cued results are the paper's,
# unchanged. Data loading/aggregation copied from interp-analysis.R.
#
# New vision results: eacl-rebuttal/results/interp_vision_new/main_runs/
#   (eacl-rebuttal/scripts/run_interp_vision_new.sh)
#
# please open the .Rproj file in the home dir of the repo so that the paths resolve.
# On the server: /opt/R/4.6.2/bin/Rscript eacl-rebuttal/analysis/interp-rebuttal.R

library(tidyverse)
library(lmerTest)

out_dir <- "eacl-rebuttal/results/interp_vision_new/compare"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

read_runs <- function(root, modality_override = NULL) {
  fs::dir_ls(root, recurse = TRUE, regexp = "*.csv") %>%
    keep(str_detect(., "(das|diffmean|probe|patch\\_k128)\\.csv$")) %>%
    map_df(read_csv, .id = "file", show_col_types = FALSE) %>%
    mutate(
      model = case_when(
        str_detect(file, "VL-2B") ~ "Qwen3-VL-2B",
        TRUE ~ "Qwen3-VL-4B"
      ),
      modality = if (!is.null(modality_override)) modality_override else case_when(
        str_detect(file, "vision") ~ "Vision",
        TRUE ~ "Language"
      ),
      attractors = str_extract(file, "(?<=att)(.*)(?=\\_opp)") |> as.numeric(),
      method = str_extract(file, "(?<=opp\\/)(.*)(?=\\.csv)"),
      before = base_logp_B - base_logp_A,
      after = base_intervention_logp_A - base_intervention_logp_B,
      odds = before + after,
      remove = case_when(
        model == "Qwen3-VL-2B" & layer == 28 ~ TRUE,
        model == "Qwen3-VL-4B" & layer == 36 ~ TRUE,
        TRUE ~ FALSE
      )
    ) %>%
    select(-file)
}

agg <- function(d) {
  d %>%
    filter(remove == FALSE) %>%
    group_by(split, layer, tok, model, modality, method, attractors) %>%
    summarize(
      n = n(),
      sd = sd(odds),
      mean_odds = mean(odds),
      conf = qt(1 - (0.05/2), n - 1) * sd/sqrt(n),
      .groups = "drop"
    )
}

critical <- function(a) {
  a %>%
    filter(split == "test") %>%
    mutate(
      region = case_when(
        is.na(tok) ~ "critical",
        attractors == 0 & tok == 5 ~ "critical",
        attractors == 1 & tok == 8 ~ "critical",
        attractors == 2 & tok == 11 ~ "critical",
        attractors == 3 & tok == 14 ~ "critical",
        TRUE ~ "pre-critical"
      ),
      method = factor(method, levels = c("das", "diffmean", "probe", "patch_k128"))
    ) %>%
    filter(region == "critical")
}

paper <- read_runs("results/interp/main_runs/")
new_vision <- read_runs("eacl-rebuttal/results/interp_vision_new/main_runs/", "Vision")
natural <- read_runs("results/interp/natural_runs/", "Pre-training")

paper_reg <- critical(agg(paper))
new_reg <- critical(agg(bind_rows(paper %>% filter(modality == "Language"), new_vision)))

# --- fit2 exactly as in interp-analysis.R ----------------------------------
fit_paper <- lmer(mean_odds ~ attractors + modality + model + method + (1|layer), data = paper_reg)
fit_new   <- lmer(mean_odds ~ attractors + modality + model + method + (1|layer), data = new_reg)

cat("\n===== fit2, paper data (language + OLD vision): should give modalityVision = -0.016, p = .92\n")
print(summary(fit_paper)$coefficients)
cat("\n===== fit2, language + NEW vision\n")
print(summary(fit_new)$coefficients)

bind_rows(
  as.data.frame(summary(fit_paper)$coefficients) %>% rownames_to_column("term") %>% mutate(fit = "language + old vision (paper)"),
  as.data.frame(summary(fit_new)$coefficients) %>% rownames_to_column("term") %>% mutate(fit = "language + new vision")
) %>% write_csv(file.path(out_dir, "fit2_cue_effect.csv"))

# --- best layer per model x method, chosen on real nouns (as in interp-analysis.R)
best_layer <- critical(agg(natural)) %>%
  group_by(model, method, layer) %>%
  summarize(odds = mean(mean_odds), .groups = "drop") %>%
  group_by(model, method) %>%
  filter(odds == max(odds)) %>%
  ungroup() %>%
  select(model, method, layer)

best_tab <- bind_rows(
  critical(agg(natural)) %>% mutate(condition = "Real nouns"),
  critical(agg(paper)) %>% mutate(condition = if_else(modality == "Vision", "Vision (paper, old)", "Language (paper)")),
  critical(agg(new_vision)) %>% mutate(condition = "Vision (new)")
) %>%
  inner_join(best_layer, by = c("model", "method", "layer")) %>%
  select(model, method, layer, attractors, condition, n, mean_odds, conf) %>%
  arrange(model, method, attractors, condition)

write_csv(best_tab, file.path(out_dir, "best_layer_odds.csv"))
cat("\n===== mean log odds at best layer (chosen on real nouns)\n")
best_tab %>%
  select(model, method, attractors, condition, mean_odds) %>%
  pivot_wider(names_from = condition, values_from = mean_odds) %>%
  mutate(across(where(is.numeric), ~ round(.x, 2))) %>%
  print(n = Inf, width = Inf)

# =============================================================================
# Figures: same code as analysis/interp-analysis.R (emb-interp-results-*,
# novel-vs-real-interp-*), with the NEW vision results in place of the old.
# Saved to eacl-rebuttal/figures/ under the paper's file names and sizes.
# =============================================================================
library(ggtext)
library(ggh4x)

fig_dir <- "eacl-rebuttal/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

embs_agg <- agg(bind_rows(paper %>% filter(modality == "Language"), new_vision))
natural_agg <- agg(natural) %>% select(-modality)

best_layer <- natural_agg %>%
  filter(split == "test") %>%
  mutate(
    region = case_when(
      is.na(tok) ~ "critical",
      attractors == 0 & tok == 5 ~ "critical",
      attractors == 1 & tok == 8 ~ "critical",
      attractors == 2 & tok == 11 ~ "critical",
      attractors == 3 & tok == 14 ~ "critical",
      TRUE ~ "pre-critical"
    ),
    method = factor(
      method,
      levels = c("das", "diffmean", "probe", "patch_k128")
    )
  ) %>%
  filter(region == "critical") %>%
  group_by(model, method, layer) %>%
  summarize(odds = mean(mean_odds), .groups = "drop") %>%
  group_by(model, method) %>%
  filter(odds == max(odds)) %>%
  ungroup() %>%
  mutate(keep=TRUE)

emb_plot <- function(which_model) {
  embs_agg %>%
    filter(split == "test") %>%
    mutate(
      region = case_when(
        is.na(tok) ~ "critical",
        attractors == 0 & tok == 5 ~ "critical",
        attractors == 1 & tok == 8 ~ "critical",
        attractors == 2 & tok == 11 ~ "critical",
        attractors == 3 & tok == 14 ~ "critical",
        TRUE ~ "pre-critical"
      ),
      method = factor(
        method,
        levels = c("das", "diffmean", "probe", "patch_k128")
      )
    ) %>%
    filter(region == "critical", model == which_model) %>%
    ggplot(aes(attractors, mean_odds, color = layer, fill = layer, group = layer)) +
    geom_point() +
    geom_line() + 
    geom_ribbon(aes(ymin = mean_odds-conf, ymax = mean_odds+conf), color = NA, alpha = 0.4) +
    ggh4x::facet_grid2(
      modality ~ method, 
      scales = "free", 
      independent = "y", 
      axes = "all",
      labeller = labeller(
        method = as_labeller(c(
          das = "'DAS'",
          diffmean = "'DiffMean'", 
          probe = "'Probe'",
          patch_k128 = "AtP^'*'"
        ), default = label_parsed)
      )
    ) +
    scale_y_continuous(breaks = scales::pretty_breaks(), limits = c(-1, 12)) +
    scale_color_gradient(
      low = "#9ecae1",   
      high = "#08306b",  
      breaks = seq(0, 35, by = 7), 
      limits = c(0, 35),
      guide = guide_colorbar(barheight = unit(8, "lines")),
      aesthetics = c("color", "fill")
    ) +
    theme_classic(base_size = 16, base_family = "Times") +
    theme(
      panel.grid = element_blank(),
      strip.background = element_blank(),
      strip.text.x = element_text(size = 14, lineheight = 1.2),
      plot.margin = margin(0, 0, 0, 0, "pt"),
      plot.background = element_rect(fill = "transparent", color = NA),
      panel.background = element_rect(fill = "transparent", color = NA),
      legend.background = element_rect(fill = "transparent", color = NA),
      legend.box.background = element_rect(fill = "transparent", color = NA),
      panel.border = element_blank(),
      plot.caption = element_text(size = 11, hjust = 1, color = "black")
    ) + 
    labs(
      x = "Attractors",
      y = "Avg. Odds (95% CI)",
      color = "Layer",
      fill = "Layer",
      caption = expression(paste(""^'*' * "AtP ", italic("is not a layerwise method")))
    )
}

emb_plot("Qwen3-VL-4B")
ggsave(file.path(fig_dir, "emb-interp-results-4B.pdf"), height = 4.29, width = 8.63, dpi = 300)
emb_plot("Qwen3-VL-2B")
ggsave(file.path(fig_dir, "emb-interp-results-2B.pdf"), height = 4.08, width = 8.66, dpi = 300)

novel_vs_real_plot <- function(which_model) {
  bind_rows(
    natural_agg %>% mutate(exp = "Real", modality = "Pre-training"),
    embs_agg %>% mutate(exp = "Novel")
  ) %>%
    filter(split == "test") %>%
    mutate(
      region = case_when(
        is.na(tok) ~ "critical",
        attractors == 0 & tok == 5 ~ "critical",
        attractors == 1 & tok == 8 ~ "critical",
        attractors == 2 & tok == 11 ~ "critical",
        attractors == 3 & tok == 14 ~ "critical",
        TRUE ~ "pre-critical"
      ),
      method = factor(
        method,
        levels = c("das", "diffmean", "probe", "patch_k128")
      )
    ) %>%
    filter(region == "critical", model == which_model) %>%
    inner_join(best_layer) %>%
    mutate(
      method = factor(
        method,
        levels = c("das", "diffmean", "probe", "patch_k128"),
        labels = c("DAS", "DiffMean", "Probe", "AtP")
      )
    ) %>%
    ggplot(aes(attractors, mean_odds, color = modality, shape = exp, fill = modality, linetype = exp)) +
    geom_point() +
    geom_line() +
    geom_ribbon(aes(ymin = mean_odds-conf, ymax = mean_odds+conf), color = NA, alpha = 0.4) +
    scale_color_manual(
      name = "Cue Condition",
      values = c("Pre-training" = "black", "Language" = "#5e3c99", "Vision" = "#e66101"),
      breaks = c("Language", "Vision")
    ) +
    scale_fill_manual(
      name = "Cue Condition",
      values = c("Pre-training" = "black", "Language" = "#5e3c99", "Vision" = "#e66101"),
      breaks = c("Language", "Vision")
    ) +
    scale_y_continuous(limits = c(-1, 12), breaks = c(0, 5, 10)) +
    facet_wrap(~method, nrow = 1, scales = "free") +
    theme_classic(base_size = 16, base_family = "Times") +
    theme(
      panel.grid = element_blank(),
      strip.background = element_blank(),
      strip.text.x = ggtext::element_markdown(size = 14, lineheight = 1.2),
      plot.margin = margin(0, 0, 0, 0, "pt"),
      plot.background = element_rect(fill = "transparent", color = NA),
      panel.background = element_rect(fill = "transparent", color = NA),
      legend.background = element_rect(fill = "transparent", color = NA),
      legend.box.background = element_rect(fill = "transparent", color = NA),
      panel.border = element_blank(),
      plot.caption = element_text(size = 11, hjust = 1, color = "black")
    ) + 
    labs(
      x = "Attractors",
      y = "Avg. Odds (95% CI)",
      color = "Cue Condition",
      fill = "Cue Condition",
      shape = "Word Type",
      linetype = "Word Type"
    )
}

novel_vs_real_plot("Qwen3-VL-4B")
ggsave(file.path(fig_dir, "novel-vs-real-interp-4B.pdf"), height = 2.34, width = 8.88, dpi = 300)
novel_vs_real_plot("Qwen3-VL-2B")
ggsave(file.path(fig_dir, "novel-vs-real-interp-2B.pdf"), height = 2.34, width = 8.88, dpi = 300)

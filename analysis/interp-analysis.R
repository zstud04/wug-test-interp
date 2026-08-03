library(tidyverse)
library(ggtext)
library(ggh4x)
library(lmerTest)

methods <- c("das", "diffmean", "probe", "patch_k128")

natural <- fs::dir_ls("results/interp/natural_runs/", recurse = TRUE, regexp = "*.csv") %>%
  keep(str_detect(., "(das|diffmean|probe|patch\\_k128)")) %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    model = case_when(
      str_detect(file, "VL-2B") ~ "Qwen3-VL-2B",
      TRUE ~ "Qwen3-VL-4B"
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

natural_agg <- natural %>%
  filter(remove == FALSE) %>%
  group_by(split, layer, tok, model, method, attractors) %>%
  summarize(
    n = n(),
    sd = sd(odds),
    mean_odds = mean(odds),
    conf = qt(1 - (0.05/2), n - 1) * sd/sqrt(n),
    .groups = "drop"
  )

natural_agg %>%
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
      levels = c("das", "diffmean", "probe", "patch_k128"),
      # Use <br> for newline and HTML <span> to reduce the font size of the subtitle
      labels = c("DAS", "DiffMean", "Probe", "AtP<sup>*</sup>")
    )
  ) %>%
  filter(region == "critical", model == "Qwen3-VL-2B") %>%
  ggplot(aes(attractors, mean_odds, color = layer, fill = layer, group = layer)) +
  geom_point() +
  geom_line() + 
  geom_ribbon(aes(ymin = mean_odds-conf, ymax = mean_odds+conf), color = NA, alpha = 0.4) +
  facet_wrap(~ method, nrow = 1, scales = "free") +
  scale_y_continuous(breaks = scales::pretty_breaks(), limits = c(-1, 12)) +
  # Viridis scale: perceptually uniform. direction = -1 makes higher layers darker.
  # scale_color_viridis_c(
  #   aesthetics = c("color", "fill"),
  #   direction = -1,
  #   breaks = seq(0, 35, by = 7), # Cleaner, evenly spaced breaks: 0, 7, 14, 21, 28, 35
  #   limits = c(0, 35),
  #   guide = guide_colorbar(barheight = 10) # Taller legend bar so breaks don't crowd
  # ) +
  scale_color_gradient(
    low = "#9ecae1",   # Lighter shade for lower layers 
    high = "#08306b",  # Darker shade for higher layers
    breaks = seq(0, 35, by = 7), 
    limits = c(0, 35),
    guide = guide_colorbar(barheight = 8),
    aesthetics = c("color", "fill")
  ) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    strip.background = element_blank(),
    # Use element_markdown to render the HTML added to the AtP label
    strip.text.x = ggtext::element_markdown(size = 14, lineheight = 1.2),
    # legend.box.spacing = unit(0, "pt"),
    plot.margin = margin(0, 0, 0, 0, "pt"),
    
    # --- New code for transparency and no borders ---
    plot.background = element_rect(fill = "transparent", color = NA), # Transparent canvas, no border
    panel.background = element_rect(fill = "transparent", color = NA), # Transparent plot area
    legend.background = element_rect(fill = "transparent", color = NA), # Transparent legend
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

# ggsave("figures/real-interp-results-4B.pdf", height = 2.68, width = 8.60, dpi = 300)
ggsave("figures/real-interp-results-2B.pdf", height = 2.68, width = 8.60, dpi = 300)



embs <- fs::dir_ls("results/interp/main_runs/", recurse = TRUE, regexp = "*.csv") %>%
  keep(str_detect(., "(das|diffmean|probe|patch\\_k128)")) %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    model = case_when(
      str_detect(file, "VL-2B") ~ "Qwen3-VL-2B",
      TRUE ~ "Qwen3-VL-4B"
    ),
    modality = case_when(
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

embs_agg <- embs %>%
  filter(remove == FALSE) %>%
  group_by(split, layer, tok, model, modality, method, attractors) %>%
  summarize(
    n = n(),
    sd = sd(odds),
    mean_odds = mean(odds),
    conf = qt(1 - (0.05/2), n - 1) * sd/sqrt(n),
    .groups = "drop"
  )

embs_agg_reg <- embs_agg %>%
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
      levels = c("das", "diffmean", "probe", "patch_k128"),
      # # Use <br> for newline and HTML <span> to reduce the font size of the subtitle
      # labels = c("DAS", "DiffMean", "Probe", "AtP<br><span style='font-size: 10pt;'>(<i>not a layerwise method</i>)</span>")
    )
  ) %>%
  filter(
    region == "critical", 
    # str_detect(model, "4B")
  )

emb_results <- embs %>%
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
      levels = c("das", "diffmean", "probe", "patch_k128"),
      # # Use <br> for newline and HTML <span> to reduce the font size of the subtitle
      # labels = c("DAS", "DiffMean", "Probe", "AtP<br><span style='font-size: 10pt;'>(<i>not a layerwise method</i>)</span>")
    )
  ) %>%
  filter(region == "critical", str_detect(model, "4B")) %>%
  select(layer, source_input, method, attractors, modality, odds) %>%
  group_by(source_input) %>%
  mutate(item_id = row_number()) %>%
  ungroup()

fit <- lmer(odds ~ layer * method + modality + (1|item_id:attractors), data = emb_results %>% filter(!str_detect(method, "patch")))

summary(fit)

fit2 <- lmer(mean_odds ~ attractors + modality + model + method + (1|layer), data = embs_agg_reg)
summary(fit2)


# EMB result plot (2B and 4B -- make changes and save acc.)

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
  filter(region == "critical", model == "Qwen3-VL-2B") %>%
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
        # Added italic() right inside scriptstyle()
        # patch_k128 = "atop('AtP', scriptstyle(italic('(not a layerwise method)')))"
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
    # legend.box.spacing = unit(0, "pt"),
    plot.margin = margin(0, 0, 0, 0, "pt"),
    
    # --- New code for transparency and no borders ---
    plot.background = element_rect(fill = "transparent", color = NA), # Transparent canvas, no border
    panel.background = element_rect(fill = "transparent", color = NA), # Transparent plot area
    legend.background = element_rect(fill = "transparent", color = NA), # Transparent legend
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

# ggsave("figures/emb-interp-results-4B.pdf", height = 4.29, width = 8.63, dpi = 300)
ggsave("figures/emb-interp-results-2B.pdf", height = 4.08, width = 8.66, dpi = 300)

# select and report best:

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
  filter(region == "critical", model == "Qwen3-VL-4B") %>%
  inner_join(best_layer) %>%
  mutate(
    method = factor(
      method,
      levels = c("das", "diffmean", "probe", "patch_k128"),
      # Use <br> for newline and HTML <span> to reduce the font size of the subtitle
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
    breaks = c("Language", "Vision") # This line hides "pre-training" from the legend
  ) +
  scale_fill_manual(
    name = "Cue Condition",
    values = c("Pre-training" = "black", "Language" = "#5e3c99", "Vision" = "#e66101"),
    breaks = c("Language", "Vision") # This line hides "pre-training" from the legend
  ) +
  scale_y_continuous(limits = c(-1, 12), breaks = c(0, 5, 10)) +
  facet_wrap(~method, nrow = 1, scales = "free") +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    strip.background = element_blank(),
    # Use element_markdown to render the HTML added to the AtP label
    strip.text.x = ggtext::element_markdown(size = 14, lineheight = 1.2),
    # legend.box.spacing = unit(0, "pt"),
    plot.margin = margin(0, 0, 0, 0, "pt"),
    
    # --- New code for transparency and no borders ---
    plot.background = element_rect(fill = "transparent", color = NA), # Transparent canvas, no border
    panel.background = element_rect(fill = "transparent", color = NA), # Transparent plot area
    legend.background = element_rect(fill = "transparent", color = NA), # Transparent legend
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
    linetype = "Word Type",
    # caption = expression(paste(""^'*' * "AtP ", italic("is not a layerwise method")))
  )

ggsave("figures/novel-vs-real-interp-4B.pdf", height = 2.34, width = 8.88, dpi = 300)
# ggsave("figures/novel-vs-real-interp-2B.pdf", height = 2.34, width = 8.88, dpi = 300)

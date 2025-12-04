library(ggplot2)
library(ggmap)

### File path (recommended for supplementary scripts)
# The dataset is expected to be located in "./data/map/sampling.txt"
data_path <- file.path("data", "map", "sampling.txt")

if (!file.exists(data_path)) {
    stop(paste0(
        "Input file not found at: ", data_path, "\n",
        "Please place 'sampling.txt' under the directory './data/map/'."
    ))
}

### Load sampling metadata
data <- read.delim(
    data_path,
    row.names = 1,
    sep = '\t',
    stringsAsFactors = FALSE,
    check.names = FALSE,
    na.strings = "na"
)

### Initialize base world map
mp <- NULL
world_map <- borders("world", colour = NA, fill = "gray80")  # base world polygon
mp <- ggplot() +
    world_map +
    ylim(-90, 90) +
    theme(
        panel.background = element_rect(color = NA, fill = 'transparent'),
        legend.key = element_rect(fill = 'transparent')
    )

### Add sampling points
mp2 <- mp +
    geom_point(
        aes(
            x = data$Longitude,
            y = data$Latitude,
            size = data$number,
            color = data$Habitat
        ),
        alpha = 0.8
    ) +
    scale_color_manual(values = c(
        "#7EAECE", "#1D4890", "#ED9628",
        "#9A2624", "#F4B295", "#157038"
    )) +
    scale_size(range = c(2, 15))

### Remove axes and legend, finalize figure
mp3 <- mp2 +
    theme(legend.position = "none") +
    xlab(NULL) +
    ylab(NULL) +
    theme(
        axis.ticks = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_blank()
    )

### Display final map
mp3

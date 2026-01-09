@echo off
REM v8: Memory-map approach - spi_flash_mmap (no copy, no cache issues)
esptool.py --chip esp32s3 --port %1 --baud 460800 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_freq 80m --flash_size 16MB 0x0 bootloader.bin 0x8000 partition-table.bin 0x10000 grape_leaf_detect.bin 0x310000 espdet_pico_320_320_grape_leaf.espdl

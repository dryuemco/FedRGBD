# D435if firmware update file (prereg 13.3, 2026-10-09)

- Version: 5.17.0.10 (production), for the D435if on node_a (S/N 239722070442).
- Source: the official RealSense page "Firmware releases D400",
  https://dev.realsenseai.com/docs/firmware-releases-d400/ , entry 5.17.0.10, Download ID 41896.
  The author downloaded it in a browser after accepting the RealSense license agreement
  (the download URL https://dev.realsenseai.com/firmware-download/?file=41896 is an
  inference from the other entries on that page; the page does not state it).
- **No published checksum exists**; the SHA256 values below were computed by us.
- Zip as downloaded: d400_series_production_fw_5_17_0_10-1.zip, 984135 bytes,
  SHA256 75231556926306e514292ef7490f002d34adc812271d24f73e102446a1ee0458
- Contents: Signed_Image_UVC_5_17_0_10.bin (1573660 bytes, zip date 2025-07-28 13:19:46),
  Legal_Notices/RealSense D400 series Firmware - Header.pdf, RealSense-D400-Series-Spec-Update-041.pdf.
- Firmware image: Signed_Image_UVC_5_17_0_10.bin, 1573660 bytes,
  SHA256 a7e10cf7b011929df2bfc6347d87471372621cc937c96cb529c3846f26b60f1c
  (identical on the desktop and on node_a, ~/fw/).
- The binary is not in this repository.

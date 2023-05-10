# -*- coding: utf-8 -*-
"""
/***************************************************************************
 QLSU
                                 A QGIS plugin
 This plugin allows linear spectral unmixing of multispectral RS data
                              -------------------
        copyright            : (C) 2022 by Bahadir Celik
        email                : bahadircelik@osmaniye.edu.tr
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

from qgis.core import (
    Qgis, 
    QgsProject,
    QgsMapLayerProxyModel, 
    QgsRasterLayer,
    QgsMessageLog
    )
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import (
    Qt,
    QSettings, 
    QTranslator, 
    QCoreApplication,
    QThread,
    pyqtSignal,
    pyqtSlot
    )
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (
    QAction, 
    QFileDialog, 
    QTableWidgetItem,
    QListWidget,
    QListWidgetItem,
    QGraphicsScene, 
    QAbstractItemView,
    QMessageBox,
    QMenu
    )
# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .qlsu_dialog import QLSUDialog
from .qlsu_spectral_resampling_dialog import (
    SpectralResamplingDialog
    )
from .qlsu_custom_spectra_dialog import (
    CustomSpectraDialog
    )
from .qlsu_plot_spectra_dialog import (
    PlotSpectraDialog
)
import os.path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
    )
try:
    import gdal
    import osr
except ImportError:
    from osgeo import gdal, osr
import os
import csv
import json
import numpy as np
from scipy.optimize import nnls
import math
import string
from .instrument_rsr import InstrumentRSR


class UnmixingTask(QThread):
    """ A Worker class for unmixing task using QThread in order to
    seperate it from main thread and keep QGIS and QLSU plugin GUI 
    responsive while performing heavy raster processing """
    
    proc       = pyqtSignal(str) # Signal to report ongoing process in console
    prog       = pyqtSignal(int) # Signal to report ongoing process in progress bar
    finished   = pyqtSignal() # Signal to report if process is finished

    def __init__(
        self, ds_img, e_data, e_names, 
        unmix_alg, constraint, out_dir, errorband
        ):
        """ Constructor.

        :param df_img: Input GeoTIFF Multi-band remote sensing imagery
        :type df_img: GDALDataset

        :param e_data: 2D Endmember data matrix (m x n)
        :type e_data: numpy array

        :param e_names: List of endmember names (n x 1)
        :type e_names: list

        :param unmix_alg: Type of least-squares unmixing algorithm
            ('Unconstrained Least Squares',
            'Partially Constrained Least Squares',
            'Fully Constrained Least Squares')
        :type unmix_alg: str

        :param constraint: Type of constraint for least-squares unmixing
            - 'SCLS'   Abundance sum-to-one constraint
            - 'NNCLS'  Abundance non-negativity constraint
        :type constraint: str

        :param out_dir: Directory for output abundance and error images
        :type out_dir: str

        :param errorband: Option for computing error band
        :type errorband: bool 
        """

        super(QThread, self).__init__()
        self.ds_img      = ds_img
        self.e_data      = e_data
        self.e_names     = e_names
        self.out_dir     = out_dir
        self.unmix_alg   = unmix_alg
        self.constraint  = constraint
        self.errorband   = errorband
        self.stoptask    = False
        self.e_datapinv  = np.linalg.pinv(self.e_data)
        self.invete      = np.linalg.inv(np.dot(self.e_data.T, self.e_data))
        self.onescls     = np.full((self.e_data.shape[1], 1),1)
        self.onestinvete = np.linalg.inv(np.dot(np.dot(self.onescls.T, self.invete), self.onescls))
        self.onesfcls    = np.array(np.full((1, self.e_data.shape[1]), 1))
        self.M           = np.append(self.e_data, self.onesfcls, axis=0)

    def UCLS(self, bline):
        """ Performs Unconstrained Least Squares Unmixing
            
        :param bline: Pixel vector to be processed (m x 1)
        :type bline: list

        :returns: Abundance vector (n x 1)
        :rtype: numpy array 
        """

        # e_pinv = np.linalg.pinv(self.e_data)
        # abund = np.dot(np.linalg.pinv(self.e_data), np.array(bline))
        abund = np.dot(self.e_datapinv, np.array(bline))
        return abund
    
    def SCLS(self, bline):
        """ Performs Sum-to-one Constrained Least Squares Unmixing
            
        :param bline: Pixel vector to be processed (m x 1)
        :type bline: list

        :returns: Abundance vector (n x 1)
        :rtype: numpy array 
        """

        abunducls = self.UCLS(bline)
        #invete = np.linalg.inv(np.dot(self.e_data.T, self.e_data))
        #ones = np.full((self.e_data.shape[1], 1),1)
        lampd = 1 - (np.dot(self.onescls.T, abunducls))
        #onestinvete = np.linalg.inv(np.dot(np.dot(ones.T, invete), ones))
        lam = np.dot(self.onestinvete, lampd)
        abund = abunducls + (lam * np.dot(self.invete, self.onescls).flatten())
        return abund


    def NNCLS(self, bline):
        """ Performs Non-negativity Constrained Least Squares Unmixing
            
        :param bline: Pixel vector to be processed (m x 1)
        :type bline: list

        :returns: Abundance vector (n x 1)
        :rtype: numpy array 
        """

        abund = nnls(self.e_data, bline)[0]
        return abund

    def FCLS(self, bline):
        """ Performs Fully Constrained Least Squares Unmixing
            
        :param bline: Pixel vector to be processed (m x 1)
        :type bline: list

        :returns: Abundance vector (n x 1)
        :rtype: numpy array 
        """

        # ones = np.array(np.full((1, self.e_data.shape[1]), 1))
        # M = np.append(self.e_data, self.onesfcls, axis=0)
        b = np.append(bline, 1)
        abund = nnls(self.M, b)[0]
        return abund

    def split_chunks(self):
        chunksize = 256 * 256

        reschunk = self.totalprog %  chunksize
        numchunks = self.totalprog // chunksize

        chunks = []
        for i in range(numchunks):
            if (i < numchunks - 1):
                chunks.append((i * chunksize, (i+1) * chunksize))
            else:
                chunks.append((i*chunksize, (i+1)*chunksize + reschunk))
        return chunks

    
    def compute_rmse(self, refl_vector, abund_vector):
        """ Compute root mean square error

        :param refl_vector: Input reflectance vector for pixel (1 x m)
        :type refl_vector: list

        :param e_data: 2D Endmember data matrix (m x n)
        :type e_data: numpy array

        :param abund_vector: Abundance vector for pixel (1 x n)
        :type abund_vector: list

        :returns: Root mean square error for pixel reflectance values
        :rtype: float """

        # refl_unmix = []
        # for i in range(self.e_data.shape[0]):
        #     refl = np.sum(np.multiply(abund_vector, self.e_data[i]))
        #     refl_unmix.append(refl)
        refl_unmix = np.dot(self.e_data, abund_vector)
        rmse = np.sqrt(np.square(np.subtract(refl_vector, refl_unmix)).mean())
        return rmse


    def run(self):
        """ Run method for UnmixingTask Class  """

        # Initialize values for prog and proc signals
        self.prog.emit(0) # Progress bar to zero
        self.proc.emit(" ") # Console process information to empty string

        # Get input raster row and columnt count
        self.rowcount    = self.ds_img.RasterYSize
        self.columncount = self.ds_img.RasterXSize

        # Total image size to compute progress information
        self.totalprog = self.rowcount * self.columncount


        # Read input image and get info
        inputdata = self.ds_img.ReadAsArray()
        self.bandcount = self.ds_img.RasterCount
        input_nodata = self.ds_img.GetRasterBand(1).GetNoDataValue()

        if input_nodata == None:
            input_nodata = 0.0000001

        input_driver = self.ds_img.GetDriver()
        input_geotransform = self.ds_img.GetGeoTransform()

        # Create output abundance datasets
        outputds = []
        for n in self.e_names:
            self.proc.emit("Processing output images.")
            ds_abundance = input_driver.Create(
                utf8_path = os.path.join(self.out_dir, n) + ".tif", 
                xsize = self.columncount,
                ysize = self.rowcount,
                bands = 1,
                eType = gdal.GDT_Float32
            )
            ds_abundance.SetGeoTransform(input_geotransform)
            srs_abundance = osr.SpatialReference()
            srs_abundance.ImportFromWkt(self.ds_img.GetProjectionRef())
            ds_abundance.SetProjection(srs_abundance.ExportToWkt())
            outputds.append(ds_abundance)
        
        # Create output error band dataset
        if self.errorband == True:
            ds_errorband = input_driver.Create(
                utf8_path = os.path.join(self.out_dir, "Rmse.tif"),
                xsize = self.columncount,
                ysize = self.rowcount,
                bands = 1,
                eType = gdal.GDT_Float32
            )
            ds_errorband.SetGeoTransform(input_geotransform)
            srs_errorband = osr.SpatialReference()
            srs_errorband.ImportFromWkt(self.ds_img.GetProjectionRef())
            ds_errorband.SetProjection(srs_errorband.ExportToWkt())
            self.errorband_arr  = np.zeros((self.totalprog))
       
        self.abundbands_arr = np.zeros((self.totalprog, len(self.e_names)))

        # Split image into chunks
        chunks = self.split_chunks()

        # Vectorize mixing functions
        ucls_v  = np.vectorize(self.UCLS,  signature="(n)->(p)")
        fcls_v  = np.vectorize(self.FCLS,  signature="(n)->(p)")
        nncls_v = np.vectorize(self.NNCLS, signature="(n)->(p)")
        scls_v  = np.vectorize(self.SCLS,  signature="(n)->(p)")

        # Vectorize rmse function
        rmse_v  = np.vectorize(self.compute_rmse, signature="(n),(p)->()")

        # Stack data
        self.stacked = np.vstack(np.dstack(inputdata))
        inputdata = None

        # Start processing
        for i,j in chunks:
            if self.stoptask == True:
                QgsMessageLog.logMessage(
                    "Process cancelled.",
                    "QLSU", Qgis.Info)
                break
            if self.unmix_alg == "Unconstrained Least Squares":
                self.abundbands_arr[i:j] = ucls_v(self.stacked[i:j])
                proginfo = int((j / self.totalprog) * 100)
                self.prog.emit(proginfo)
            if  self.unmix_alg == "Partially Constrained Least Squares":
                if self.constraint == "SCLS":
                    self.abundbands_arr[i:j] = scls_v(self.stacked[i:j])
                if self.constraint == "NNCLS":
                    self.abundbands_arr[i:j] = nncls_v(self.stacked[i:j])
                proginfo = int((j / self.totalprog) * 100)
                self.prog.emit(proginfo)
            if self.unmix_alg == "Fully Constrained Least Squares":
                self.abundbands_arr[i:j] = fcls_v(self.stacked[i:j])
                proginfo = int((j / self.totalprog) * 100)
                self.prog.emit(proginfo)
        
        # Compute RMSE Image
        if self.errorband == True:
            self.proc.emit("Calculating RMSE Image.")
            self.prog.emit(0)
            for i,j in chunks:
                if self.stoptask == True:
                    QgsMessageLog.logMessage(
                        "Process cancelled.",
                        "QLSU", Qgis.Info)
                self.errorband_arr[i:j] = rmse_v(self.stacked[i:j], self.abundbands_arr[i:j])
                proginfo = int((j / self.totalprog) * 100)
                self.prog.emit(proginfo)

        # Clear stacked image array
        self.stacked = None
        abundance = np.reshape(self.abundbands_arr.T, (len(self.e_names), self.rowcount, self.columncount))
        if self.errorband == True:
            rmse =  np.reshape(self.errorband_arr.T, (self.rowcount, self.columncount))

        # Write abundance images to file
        for i in range(len(self.e_names)):
            outputds[i].GetRasterBand(1).WriteArray(abundance[i], 0, 0)
            outputds[i].GetRasterBand(1).FlushCache()
        
        # Write RMSE image to file
        if self.errorband == True:
            ds_errorband.GetRasterBand(1).WriteArray(rmse)
            ds_errorband.GetRasterBand(1).FlushCache()

        if self.stoptask == True:
            QgsMessageLog.logMessage(
                "Process cancelled.",
                "QLSU", Qgis.Info)

        # Clear datasets and arrays
        for ds in outputds:
            ds = None
        outputds = None
        abundance = None
        self.abundbands_arr = None
        self.ds_img = None
        ds_abundance = None

        if self.errorband == True:
            ds_errorband = None
            self.errorband_arr = None
            rmse = None

        self.proc.emit("Done!")
        self.finished.emit()

    def stop(self):
        """ Sets unmixing process stop indicator """

        self.stoptask = True

class QLSU:
    """QGIS Plugin Implementation. """

    def __init__(self, iface):
        """ Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """

        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'QLSU_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.menu = self.tr(u'&QLSU')

        self.toolbar = self.iface.addToolBar(u'QLSU')
        self.toolbar.setObjectName(u'QLSU')

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """ Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """

        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('QLSU', message)

    def initGui(self):
        """ Create the menu entries and toolbar icons inside the QGIS GUI."""
        
        self.toolbar = self.iface.addToolBar(u'QLSU')
        self.toolbar.setObjectName(u'QLSUToolBar')
        self.toolbar_unmixing = QAction(
            QIcon(':/plugins/qlsu/icons/icon.png'),
            self.tr(u'Linear Spectral Unmixing'),
            self.iface.mainWindow()
            )
        self.toolbar_unmixing.setObjectName('toolbar_unmixing')
        self.toolbar_unmixing.setCheckable(False)
        self.toolbar_unmixing.triggered.connect(self.show_toolbar_unmixing)
        self.toolbar.addAction(self.toolbar_unmixing)

        self.toolbar_spectralresampling = QAction(
            QIcon(':/plugins/qlsu/icons/icon_sr.png'),
            self.tr(u'Spectral Library Viewer and Resampling'),
            self.iface.mainWindow()
        )
        self.toolbar_spectralresampling.setObjectName('toolbar_spectralresampling')
        self.toolbar_spectralresampling.setCheckable(False)
        self.toolbar_spectralresampling.triggered.connect(self.show_toolbar_spectralresampling)
        self.toolbar.addAction(self.toolbar_spectralresampling)

        self.iface.addPluginToRasterMenu(u'&QLSU', self.toolbar_unmixing)
        self.iface.addPluginToRasterMenu(u'&QLSU', self.toolbar_spectralresampling)

        # will be set False in run()
        self.first_start = True

    def show_toolbar_unmixing(self):
        """ Show spectral unmixing dialog """

        self.dlg  = QLSUDialog()
        self.dlg.radioButtonSCLS.setChecked(True)
        self.dlg.radioButtonSCLS.setHidden(True)
        self.dlg.radioButtonNNCLS.setHidden(True)
        self.dlg.mMapLayerComboBox.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.dlg.comboBoxAlgorithm.clear()
        # Set spectral unmixing algorithm for comboBoxAlgoritm
        unmixingoptions = ["Unconstrained Least Squares",
        "Partially Constrained Least Squares", "Fully Constrained Least Squares"]
        for option in unmixingoptions:
            self.dlg.comboBoxAlgorithm.addItem(option)
        
        self.dlg.lineEditEndmemberPath.clear()
        self.dlg.pushButtonBrowseSpectra.clicked.connect(self.browse_endmember_spectra)
        self.dlg.pushButtonOutput.clicked.connect(self.browse_output_dir)
        self.dlg.pushButtonLoadSpectra.clicked.connect(self.load_endmembers)
        self.dlg.comboBoxAlgorithm.currentTextChanged.connect(self.toggle_parameters)
        self.dlg.browseImageButton.clicked.connect(self.load_image)
        self.dlg.pushButtonRun.clicked.connect(self.start_unmixing_task)
        self.dlg.pushButtonCancel.clicked.connect(self.stop_unmixing_task)
        self.dlg.pushButtonQuitTool.clicked.connect(self.dlg.close)

        # show the dialog
        self.dlg.show()
        self.dlg.exec_()

    def parseECOSTRESS(self, ecostressfile):
        """ Parses ECOSTRESS spectral library file
        
        :param ecostressfile: ECOSTRESS spectral library file path
        :type ecostressfile: str

        :returns: Python dictionary containing spectral data of materials
        :rtype: dict
        """

        spectra_data = {}
        with open(ecostressfile, encoding='utf-8') as ecostress_data:
            line_data = ecostress_data.readlines()
            ecostress_metadata = {}
        # Parse metadata
        for line in line_data:
            metadata_list = line.strip().split(":")
            try:
                float(metadata_list[1].strip())
                ecostress_metadata[metadata_list[0].strip()] = float(metadata_list[1].strip())
            except IndexError:
                self.message = QMessageBox.warning(
                    None, "Error", "Not a valid ECOSTRESS library data!", 
                    QMessageBox.Ok, QMessageBox.Ok)
                return
            except ValueError:
                if metadata_list[0].strip() == "Name":
                    ecostress_metadata[metadata_list[0].strip()] = metadata_list[1].strip()
                if metadata_list[0].strip() == "Class":
                    ecostress_metadata[metadata_list[0].strip()] = metadata_list[1].strip()
                if metadata_list[0].strip() == "Sample No.":
                    ecostress_metadata[metadata_list[0].strip()] = metadata_list[1].strip()
                # if metadata_list[0].strip() == "Description":
                #     ecostress_metadata[metadata_list[0].strip()] = metadata_list[1].strip()
                if metadata_list[0].strip() == "X Units":
                    ecostress_metadata[metadata_list[0].strip()] = metadata_list[1].strip()
                if metadata_list[0].strip() == "Y Units":
                    ecostress_metadata[metadata_list[0].strip()] = metadata_list[1].strip()
            if metadata_list[0] == "Additional Information":
                break
        
        # Parse spectra
        data = {}
        datawlenlist = []
        dataspectralist = []
        for line in line_data:
            if len(line.strip().split(":"))==2:
                continue
            # Empty line before the actual spectra
            elif len(line.strip()) == 0:
                continue
            else:
                metadata_list = line.strip().split()
                if (ecostress_metadata["X Units"] == "Wavelength (micrometers)" or 
                    ecostress_metadata["X Units"] == "Wavelength (micrometer)"):
                    datawlenlist.append(float(metadata_list[0].strip()))
                elif ecostress_metadata["X Units"] == "Wavelength (nanometers)":
                    datawlenlist.append(float(metadata_list[0].strip())/1000)
                if (ecostress_metadata["Y Units"] == "Reflectance (percent)" or 
                    ecostress_metadata["Y Units"] == "Reflectance (percentage)"):
                    dataspectralist.append(float(metadata_list[1].strip())/100)
                elif ecostress_metadata["Y Units"] != "Reflectance (percent)":
                    dataspectralist.append(float(metadata_list[1].strip()))

        ecostress_metadata["X Units"] = "Micrometers"
        ecostress_metadata["Y Units"] = "Reflectance"
        data["wlen"] = datawlenlist
        data["refl"] = dataspectralist
        spectra_data["metadata"] = ecostress_metadata
        spectra_data["spectra"] = data
        return spectra_data
    
    def spectra_to_dict(self, spectra_data):
        """ Rearrange spectra dict for listWidget

        :param spectra_data: Python dictionary containing spectral data of materials
        :type spectra_data: dict
        """

        self.spectra_dict[
            spectra_data["metadata"]["Sample No."] + " (" +
            spectra_data["metadata"]["Name"] + ")"
            ] = spectra_data

    def load_spectra_list(self, spectra_dict):
        """ Loads spectra to listWidget 
        
        :param spectra_data: Python dictionary containing spectral data of materials
        :type spectra_data: dict
        """

        self.dlg2.listWidgetSpectra.clear()
        for key in spectra_dict.keys():
            item = QListWidgetItem()
            item.setText(str(key))
            self.dlg2.listWidgetSpectra.addItem(item)
            item.setData(Qt.UserRole, spectra_dict[key])
    
    def remove_spectra(self):
        """ Removes spectra from listWidget """

        for item in self.dlg2.listWidgetSpectra.selectedItems():
            self.dlg2.listWidgetSpectra.takeItem(self.dlg2.listWidgetSpectra.row(item))
            self.spectra_dict.pop(item.text())
    
    def save_imported_spectra(self):
        """ Saves selected listWidget spectra as JSON file """

        spectra_to_save = {}
        if len(self.dlg2.listWidgetSpectra.selectedItems()) == 0:
            self.message = QMessageBox.warning(
                None, "Warning", "No spectra is selected!", 
                QMessageBox.Ok, QMessageBox.Ok)
            return
        elif len(self.dlg2.listWidgetSpectra.selectedItems()) > 0:
            for item in self.dlg2.listWidgetSpectra.selectedItems():
                spectra_to_save[str(item.text())] = item.data(QtCore.Qt.UserRole)
        save_json_dialog = QFileDialog()
        save_json_dialog.setDefaultSuffix("json")
        sfile_name, _ = save_json_dialog.getSaveFileName(
            self.dlg2, 
            "Save JSON File", 
            "", 
            "JSON (*.json)"
            )
        sfile_name = sfile_name + ".json"
        with open(sfile_name, 'w') as sf:
            json_spectra = json.dumps(spectra_to_save, sort_keys=True, indent=4)
            sf.write(json_spectra)

    def load_json_spectra(self):
        """ Loads JSON spectra file to listWidgetSpectra """

        spectra_json, _filter = QFileDialog.getOpenFileName(self.dlg2, 
        "Open JSON File (.json)",
        os.getenv('HOME'), "*.json")
        if not spectra_json:
            return
        else:
            with open(spectra_json) as sjson_file:
                spectra_json_data = json.load(sjson_file)
            for key in spectra_json_data.keys():
                self.spectra_dict[key] = spectra_json_data[key]
            self.load_spectra_list(self.spectra_dict)

    def plot_spectra(self):
        """ Plots selected spectra """

        legend_items = []
        self.dlg4 = PlotSpectraDialog(self.iface)
        fig_spectra = plt.figure()
        canvas_spectra = FigureCanvas(fig_spectra)
        toolbar_spectra = NavigationToolbar(canvas_spectra, self.dlg4)
        # Remove unnecessary subplots button from figure toolbar
        for toolbar_button in toolbar_spectra.actions():
            if toolbar_button.text() == "Subplots":
                toolbar_spectra.removeAction(toolbar_button)
        if len(self.dlg2.listWidgetSpectra.selectedItems()) == 0:
            return
        elif len(self.dlg2.listWidgetSpectra.selectedItems()) > 0:
            if len(self.dlg2.listWidgetSpectra.selectedItems()) == 1:
                item = self.dlg2.listWidgetSpectra.selectedItems()[0]
                plt.title("Plot for " + \
                str(item.data(QtCore.Qt.UserRole)["metadata"]["Sample No."]) + \
                "(" + str(item.data(QtCore.Qt.UserRole)["metadata"]["Name"]) + ")"    
                )
            else:
                plt.title("Spectra Plot")
            for item in self.dlg2.listWidgetSpectra.selectedItems():
                legend_items.append(
                    str(item.data(QtCore.Qt.UserRole)["metadata"]["Sample No."]) + \
                    "(" + str(item.data(QtCore.Qt.UserRole)["metadata"]["Name"]) + ")"
                    )
                plt.plot(item.data(QtCore.Qt.UserRole)["spectra"]["wlen"],
                         item.data(QtCore.Qt.UserRole)["spectra"]["refl"])
            plt.legend(legend_items)
            plt.xlabel("Wavelength (Micrometers)")
            plt.ylabel("Reflectance")
            plt.grid()
            self.dlg4.verticalLayout.addWidget(canvas_spectra)
            self.dlg4.horizontalLayout.addWidget(toolbar_spectra)
            self.dlg4.horizontalLayout.setAlignment(QtCore.Qt.AlignLeft)
            canvas_spectra.draw()
            plt.close(fig_spectra)
            self.dlg4.exec_()

    def plot_resampled_spectra(self):
        """ Plots selected resampled spectra """

        sensor_list = []
        for item in self.dlg2.listWidgetResampledSpectra.selectedItems():
            sensor_list.append(item.data(QtCore.Qt.UserRole)["Sensor"])
        if all(sensor == sensor_list[0] for sensor in sensor_list) == False:
            self.message = QMessageBox.warning(
                None, "Error", "Can not create multiple plot for different types of sensors.", 
                QMessageBox.Ok, QMessageBox.Ok)
            return
        else:
            legend_items = []
            self.dlg4 = PlotSpectraDialog(self.iface)
            fig_resampled_spectra = plt.figure()
            canvas_resampled_spectra = FigureCanvas(fig_resampled_spectra)
            toolbar_resampled_spectra = NavigationToolbar(canvas_resampled_spectra, self.dlg4)
            # Remove unnecessary subplots button from figure toolbar
            for toolbar_button in toolbar_resampled_spectra.actions():
                if toolbar_button.text() == "Subplots":
                    toolbar_resampled_spectra.removeAction(toolbar_button)
            if len(self.dlg2.listWidgetResampledSpectra.selectedItems()) == 0:
                return
            elif len(self.dlg2.listWidgetResampledSpectra.selectedItems()) > 0:
                if len(self.dlg2.listWidgetResampledSpectra.selectedItems()) == 1:
                    item = self.dlg2.listWidgetResampledSpectra.selectedItems()[0]
                    plt.title("Plot for " + str(item.text()))
                else:
                    plt.title("Resampled Spectra Plot")
                for item in self.dlg2.listWidgetResampledSpectra.selectedItems():
                    legend_items.append(str(item.text()))
                    plt.plot(item.data(QtCore.Qt.UserRole)["Band Centers"],
                            item.data(QtCore.Qt.UserRole)["Reflectance"])
                plt.legend(legend_items)
                plt.xlabel("Band Center Wavelengths (Micrometers)")
                plt.ylabel("Reflectance")
                plt.grid()
                self.dlg4.verticalLayout.addWidget(canvas_resampled_spectra)
                self.dlg4.horizontalLayout.addWidget(toolbar_resampled_spectra)
                self.dlg4.horizontalLayout.setAlignment(QtCore.Qt.AlignLeft)
                canvas_resampled_spectra.draw()
                plt.close(fig_resampled_spectra)
                self.dlg4.exec_()

    def save_resampled_csv(self):
        """ Saves selected resampled spectra as CSV file """
        
        sensor_list = []
        for item in self.dlg2.listWidgetResampledSpectra.selectedItems():
            sensor_list.append(item.data(QtCore.Qt.UserRole)["Sensor"])
        if all(sensor == sensor_list[0] for sensor in sensor_list) == False:
            self.message = QMessageBox.warning(
                None, "Error", "Can not save resampled spectra for different types of sensors.", 
                QMessageBox.Ok, QMessageBox.Ok)
            return
        if len(self.dlg2.listWidgetResampledSpectra.selectedItems()) == 1:
            csv_header = ["Wavelength"]
            csv_dict_list = []
            for item in self.dlg2.listWidgetResampledSpectra.selectedItems():
                csv_header.append(item.data(QtCore.Qt.UserRole)["Sample No."])
                for i in range(len(item.data(QtCore.Qt.UserRole)["Band Centers"])):
                    csv_dict_list.append(
                        {
                            "Wavelength": item.data(QtCore.Qt.UserRole)["Band Centers"][i],
                            item.data(QtCore.Qt.UserRole)["Sample No."]:item.data(QtCore.Qt.UserRole)["Reflectance"][i]
                        })
            
            save_csv_dialog = QFileDialog()
            save_csv_dialog.setDefaultSuffix("csv")
            csvfile_name, _ = save_csv_dialog.getSaveFileName(
                self.dlg2,
                "Save resampled spectra as csv file",
                "",
                "CSV (*.csv)"
            )
            csvfile_name = csvfile_name + ".csv"
            try:
                with open(csvfile_name, 'w') as csv_resampled_file:
                    csv_spectra_writer = csv.DictWriter(
                        csv_resampled_file,
                        fieldnames=csv_header,
                        delimiter=';')
                    csv_spectra_writer.writeheader()
                    for csv_data in csv_dict_list:
                        csv_spectra_writer.writerow(csv_data)
            except IOError as e:
                self.message = QMessageBox.warning(
                    None, "Error", "Error writing csv data.", 
                QMessageBox.Ok, QMessageBox.Ok)
                return

        elif len(self.dlg2.listWidgetResampledSpectra.selectedItems()) > 0:
            csv_header = ["Wavelength"]
            csv_dict_list = []
            resampled_list = []
            wavelengths = [item.data(QtCore.Qt.UserRole)["Band Centers"] for item in self.dlg2.listWidgetResampledSpectra.selectedItems()][0]
            resampled_list.append(list(wavelengths))
            for item in self.dlg2.listWidgetResampledSpectra.selectedItems():
                csv_header.append(item.data(QtCore.Qt.UserRole)["Sample No."])
                resampled_list.append(item.data(QtCore.Qt.UserRole)["Reflectance"])
            for refl in zip(*resampled_list):
                csv_dict_list.append(dict(zip(csv_header, refl)))

            save_csv_dialog = QFileDialog()
            save_csv_dialog.setDefaultSuffix("csv")
            csvfile_name, _ = save_csv_dialog.getSaveFileName(
                self.dlg2,
                "Save resampled spectra as csv file",
                "",
                "CSV (*.csv)"
            )
            csvfile_name = csvfile_name + ".csv"
            try:
                with open(csvfile_name, 'w') as csv_resampled_file:
                    csv_spectra_writer = csv.DictWriter(
                        csv_resampled_file,
                        fieldnames=csv_header,
                        delimiter=';')
                    csv_spectra_writer.writeheader()
                    for csv_data in csv_dict_list:
                        csv_spectra_writer.writerow(csv_data)
            except IOError as e:
                self.message = QMessageBox.warning(
                    None, "Error", "Error writing csv data.", 
                QMessageBox.Ok, QMessageBox.Ok)
                return            

    def browse_custom_spectra(self):
        """ Browse for custom spectra file """

        custom_spectrafile, _filter = QFileDialog.getOpenFileName(self.dlg3, 
        "Open Custom Spectra File (.txt)",
        os.getenv('HOME'), "*.txt")
        self.dlg3.lineEditCustomFile.clear()
        self.dlg3.lineEditCustomFile.setText(custom_spectrafile)
    
    def import_custom_spectra(self):
        """ Import custom spectra file """

        if self.dlg3.lineEditSpectraName.text() == '':
            self.message = QMessageBox.warning(
                None, "Error", "Spectra name cannot be empty!", 
                QMessageBox.Ok, QMessageBox.Ok)
            return
        if self.dlg3.lineEditSpectraName.text() != '':
            check_sname = self.check_endmember_names(str(self.dlg3.lineEditSpectraName.text()))
            if not check_sname:
                self.message = QMessageBox.warning(
                    None, "Error", "Name contains invalid characters!", 
                    QMessageBox.Ok, QMessageBox.Ok)
                return
            else:
                cs_name = str(self.dlg3.lineEditSpectraName.text())
        if self.dlg3.lineEditSpectraClass.text() == '':
            self.message = QMessageBox.warning(
                None, "Error", "Class cannot be empty!", 
                QMessageBox.Ok, QMessageBox.Ok)
            return
        if self.dlg3.lineEditSpectraClass.text() != '':
            check_sclass = self.check_endmember_names(str(self.dlg3.lineEditSpectraClass.text()))
            if not check_sclass:
                self.message = QMessageBox.warning(
                    None, "Error", "Class contains invalid characters!", 
                    QMessageBox.Ok, QMessageBox.Ok)
                return
            else:
                cs_class = str(self.dlg3.lineEditSpectraClass.text())
        if self.dlg3.lineEditSampleNo.text() == '':
            self.message = QMessageBox.warning(
                None, "Error", "Sample no. cannot be empty!", 
                QMessageBox.Ok, QMessageBox.Ok)
            return
        if self.dlg3.lineEditSampleNo.text() != '':
            check_sno = self.check_endmember_names(str(self.dlg3.lineEditSampleNo.text()))
            if not check_sno:
                self.message = QMessageBox.warning(
                    None, "Error", "Sample no. contains invalid characters!", 
                    QMessageBox.Ok, QMessageBox.Ok)
                return
            else:
                cs_sno = str(self.dlg3.lineEditSampleNo.text())
        if self.dlg3.tableWidgetCustom.rowCount() == 0:
            self.message = QMessageBox.warning(
                None, "Error", "Spectra is not loaded!", 
                QMessageBox.Ok, QMessageBox.Ok)
            return
        cdata_dict = self.get_custom_spectra()
        key = cs_sno + " (" + cs_name + ")"
        custom_spectra = {
            "metadata": 
            {"Name":cs_name,
             "Class":cs_class,
             "Sample No.":cs_sno
            },
            "spectra":cdata_dict
        }
        self.spectra_dict[key] = custom_spectra
        self.load_spectra_list(self.spectra_dict)
        self.dlg3.close()

    def load_custom_spectra(self):
        """ Loads custom spectra file """

        self.dlg3.tableWidgetCustom.clear()
        if self.dlg3.lineEditCustomFile.text() == '':
            self.message = QMessageBox.warning(
                None, "Error", "No file is selected!", 
                QMessageBox.Ok, QMessageBox.Ok)
            return
        if os.path.isfile(self.dlg3.lineEditCustomFile.text()) == False:
            self.message = QMessageBox.warning(
                None, "Error", "Invalid directory.",
                QMessageBox.Ok, QMessageBox.Ok)
            return
        cdata_dict = self.get_custom_spectra()
        try:
            nrows = len(cdata_dict["wlen"])
        except TypeError:
            self.message = QMessageBox.warning(
                None, "Error", "Not a valid input file!",
                QMessageBox.Ok, QMessageBox.Ok)
            return
        ncols = 2
        t_headers = ("Wavelength", "Reflectance")
        t_data    = list(zip(cdata_dict["wlen"], cdata_dict["refl"]))
        self.dlg3.tableWidgetCustom.setRowCount(nrows)
        self.dlg3.tableWidgetCustom.setColumnCount(ncols)
        self.dlg3.tableWidgetCustom.setHorizontalHeaderLabels(t_headers)
        for i in range(nrows):
            for j in range(ncols):
                self.dlg3.tableWidgetCustom.setItem(i, j, 
                QTableWidgetItem(str(t_data[i][j])))
        self.dlg3.tableWidgetCustom.setEditTriggers(QAbstractItemView.NoEditTriggers)
    
    def get_custom_spectra(self):
        """ Get loaded custom spectra information from listWidget and
            load into Python dictionary for further processing 
        
        :returns: Python dictionary containing custom spectral data
        :rtype: dict
        """

        cdata_dict = {}
        wlen_list = []
        data_list = []
        custom_spectrafile = self.dlg3.lineEditCustomFile.text()
        with open(custom_spectrafile, encoding='utf-8') as custom_data:
            line_data = custom_data.readlines()
        for line in line_data:
            sdata = line.strip().split()
            try:
                float(sdata[0].strip())
                float(sdata[1].strip())
            except ValueError:
                self.message = QMessageBox.warning(
                    None, "Error", 
                    "Not a valid spectra data!\n" + \
                    "Please read the help documentation in order to " + \
                    "check if your file is valid input file.", 
                    QMessageBox.Ok, QMessageBox.Ok)
                return
            wlen_list.append(float(sdata[0].strip()))
            data_list.append(float(sdata[1].strip()))
        cdata_dict["wlen"] = wlen_list
        cdata_dict["refl"] = data_list
        return cdata_dict

    def custom_spectra_dialog(self):
        """ Open custom spectra import dialog """

        self.dlg3 = CustomSpectraDialog(self.iface)
        self.dlg3.pushButtonBrowseFile.clicked.connect(self.browse_custom_spectra)
        self.dlg3.pushButtonLoadCustomFile.clicked.connect(self.load_custom_spectra)
        self.dlg3.pushButtonImport.clicked.connect(self.import_custom_spectra)
        self.dlg3.exec_()

    
    def load_resampled_list(self):
        """ Loads resampled spectra to listWidget """

        widget_item_list = []
        for i in range(self.dlg2.listWidgetResampledSpectra.count()):
            widget_item_list.append(self.dlg2.listWidgetResampledSpectra.item(i).text())
        if len(widget_item_list) == 0:
            for key in self.resampled_spectra.keys():
                item = QListWidgetItem()
                item.setText(str(key))
                self.dlg2.listWidgetResampledSpectra.addItem(item)
                item.setData(Qt.UserRole, self.resampled_spectra[key])
        if len(widget_item_list) > 0:
            for key in self.resampled_spectra.keys():
                if key not in widget_item_list:
                    item = QListWidgetItem()
                    item.setText(str(key))
                    self.dlg2.listWidgetResampledSpectra.addItem(item)
                    item.setData(Qt.UserRole, self.resampled_spectra[key])

    def add_spectra_menu_trigger(self, action):
        """ Add spectra button menu trigger """

        if action.text() == "Import ECOSTRESS Library File":
            ecostressfile, _filter = QFileDialog.getOpenFileName(self.dlg2, 
            "Open ECOSTRESS Library File",
            os.getenv('HOME'), "*.txt")
            if not ecostressfile:
                return
            else:
                s = self.parseECOSTRESS(ecostressfile)
                try:
                    self.spectra_to_dict(s)
                except TypeError:
                    return
        if action.text() == "Import Custom Spectra File":
            self.custom_spectra_dialog()
        self.load_spectra_list(self.spectra_dict)
    
    def convolve_spectra(self):
        """ Convolve imported spectra to selected instrument 
            band relative response functions """
        
        instrument = self.dlg2.comboBoxInstrument.currentText()
        self.rsr = InstrumentRSR(instrument)
        bwlen = []
        for item in self.dlg2.listWidgetSpectra.selectedItems():
            for bandno in self.rsr.bands.keys():
                indice = \
                (np.array(item.data(QtCore.Qt.UserRole)["spectra"]["wlen"]) >= \
                    np.min(np.array(self.rsr.bands[bandno]["Wavelengths"]))) & \
                (np.array(item.data(QtCore.Qt.UserRole)["spectra"]["wlen"]) <= \
                    np.max(np.array(self.rsr.bands[bandno]["Wavelengths"])))
                x = np.array(item.data(QtCore.Qt.UserRole)["spectra"]["wlen"])[indice]
                y = np.array(item.data(QtCore.Qt.UserRole)["spectra"]["refl"])[indice]
                refl_intp = np.interp(np.array(self.rsr.bands[bandno]["Wavelengths"]), x, y)
                refl = np.sum(
                    np.multiply(np.array(self.rsr.bands[bandno]["Response"]), 
                    refl_intp))/np.sum(np.array(self.rsr.bands[bandno]["Response"]))
                bwlen.append(refl)
            self.resampled_spectra[
                item.data(QtCore.Qt.UserRole)["metadata"]["Sample No."] + 
                "(" + self.rsr.sensor +")"
                ] = {"Name":item.data(QtCore.Qt.UserRole)["metadata"]["Name"],
                     "Sample No.":item.data(QtCore.Qt.UserRole)["metadata"]["Sample No."],
                     "Band Centers":self.rsr.bandcenters,
                     "Reflectance":bwlen,
                     "Sensor":self.rsr.sensor
                    }
            bwlen = []
        self.load_resampled_list()

    def show_toolbar_spectralresampling(self):
        """ Show spectral resampling dialog """

        self.dlg2 = SpectralResamplingDialog(self.iface)
        self.spectra_dict = {}
        self.dlg2.comboBoxInstrument.clear()
        self.resampled_spectra = {}
        instruments = ["Landsat-5 TM", "Landsat-7 ETM+",
                       "Landsat-8 OLI", "Landsat-9 OLI-2", 
                       "Sentinel-2A MSI", "Sentinel-2B MSI"]
        for instrument in instruments:
            self.dlg2.comboBoxInstrument.addItem(instrument)
        self.dlg2.listWidgetSpectra.setSelectionMode(QListWidget.ExtendedSelection)
        self.dlg2.listWidgetResampledSpectra.setSelectionMode(QListWidget.ExtendedSelection)
        menu = QMenu(
            self.dlg2.pushButtonAddSpectra, 
            triggered=self.add_spectra_menu_trigger
        )
        for menuitem in ("Import ECOSTRESS Library File", "Import Custom Spectra File"):
            menu.addAction(menuitem)
        self.dlg2.pushButtonAddSpectra.setMenu(menu)
        self.dlg2.pushButtonResample.clicked.connect(self.convolve_spectra)
        self.dlg2.pushButtonDeleteSpectra.clicked.connect(self.remove_spectra)
        self.dlg2.pushButtonSaveSpectra.clicked.connect(self.save_imported_spectra)
        self.dlg2.pushButtonLoadJson.clicked.connect(self.load_json_spectra)
        self.dlg2.pushButtonPlotSpectra.clicked.connect(self.plot_spectra)
        self.dlg2.pushButtonPlotResampled.clicked.connect(self.plot_resampled_spectra)
        self.dlg2.pushButtonSaveResampled.clicked.connect(self.save_resampled_csv)
        self.dlg2.pushButtonQuit.clicked.connect(self.dlg2.close)
        self.dlg2.exec_()

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        
        self.iface.removePluginRasterMenu(u'&QLSU', self.toolbar_unmixing)
        self.iface.removePluginRasterMenu(u'&QLSU', self.toolbar_spectralresampling)
        self.iface.removeToolBarIcon(self.toolbar_unmixing)
        self.iface.removeToolBarIcon(self.toolbar_spectralresampling)
        del self.toolbar


    def load_image(self):
        """ Loads multispectral input imagery and adds to QGIS project layer"""

        image_file, _filter = QFileDialog.getOpenFileName(self.dlg, 
        "Load Multi-Spectral Geotiff Imagery", 
        os.getenv('HOME'), "*.tif")
        if not image_file:
            return
        else:
            from pathlib import Path
            img_filepath = Path(image_file)
            img_layer = QgsRasterLayer(image_file, img_filepath.stem)
            if not img_layer.isValid():
                self.iface.messageBar().pushMessage(
                    "Failed to load image file", 
                    level=Qgis.Critical)
            else:
                QgsProject.instance().addMapLayer(img_layer)


    def browse_endmember_spectra(self):
        """ Browse endmember CSV file """

        spectrafile, _filter = QFileDialog.getOpenFileName(self.dlg, 
        "Open Enbmember Spectra File (.csv)",
        os.getenv('HOME'), "*.csv")
        self.dlg.lineEditEndmemberPath.clear()
        self.dlg.lineEditEndmemberPath.setText(spectrafile)

    def browse_output_dir(self):
        """ Browse output abundance and error image directory """

        out_dir = str(QFileDialog.getExistingDirectory(None, "Select Ouput Directory"))
        self.dlg.lineEditOutput.clear()
        self.dlg.lineEditOutput.setText(out_dir)

    def check_endmember_names(self, endmember_name):
        """ Check endmember naming for allowed characters """

        allowedchars = " .-_~"+ string.ascii_letters + string.digits
        check_name = all([c in allowedchars for c in endmember_name])
        if check_name == True:
            return True
        elif check_name == False:
            return False

    def get_endmembers(self):
        """ Gets endmember directory from lineEdit and returns 
            endmember parameters 

        :returns:
            - endmember_names - List of endmember names
            - endmember_wlengths - Spectra wavelengths, numpy array (m x 1)
            - endmember_data - Spectra data, numpy array (m x n) """
        
        spectrafile = self.dlg.lineEditEndmemberPath.text()
        with open(spectrafile, newline='') as csvfile:
            csvdata = list(csv.reader(csvfile, delimiter=';'))
        endmember_names = csvdata[0][1:]
        for i in endmember_names:
            check_name = self.check_endmember_names(i)
            if not check_name:
                return False
        endmember_wlengths = np.array(csvdata[1:])[:,0]
        for i in endmember_wlengths:
            if i.isdigit() == True:
                endmember_wlengths = endmember_wlengths.astype(int)
            else:
                endmember_wlengths = endmember_wlengths.astype(float)
        endmember_data = np.delete(np.array(csvdata[1:], dtype=float), 0, 1)
        return endmember_names, endmember_wlengths, endmember_data


    def load_endmembers(self):
        """ Loads endmember data to tableWidget and plots spectra
            in QGraphicsScene """
        
        if self.dlg.lineEditEndmemberPath.text() == '':
            self.message = QMessageBox.warning(
                None, "Error", "No endmember spectra file is selected!", 
                QMessageBox.Ok, QMessageBox.Ok)
            return

        if os.path.isfile(self.dlg.lineEditEndmemberPath.text()) == False:
            self.message = QMessageBox.warning(
                None, "Error", "Invalid directory.", 
                QMessageBox.Ok, QMessageBox.Ok)
            return
        else:
            try:
                self.e_names, self.e_wlengths, self.e_data = self.get_endmembers()
            except TypeError:
                    self.message = QMessageBox.warning(
                    None, "Error", "One or more endmember name contain invalid characters.",
                    QMessageBox.Ok, QMessageBox.Ok)
                    return
            nrows = self.e_data.shape[0]
            ncols = self.e_data.shape[1]
            self.fig = plt.figure(figsize=(3.7, 2.4)) #(figsize=(4.2, 3))
            plt.rcParams.update({'font.size':5})
            plt.subplots_adjust(hspace = 0, wspace = 0)
            plt.title("Endmember Spectra Plot")
            self.dlg.tableWidgetEndmember.setRowCount(nrows)
            self.dlg.tableWidgetEndmember.setColumnCount(ncols)
            self.dlg.tableWidgetEndmember.setHorizontalHeaderLabels(self.e_names)
            self.dlg.tableWidgetEndmember.setVerticalHeaderLabels(self.e_wlengths.astype(str))
            for i in range(nrows):
                for j in range(ncols):
                    self.dlg.tableWidgetEndmember.setItem(i, j, 
                    QTableWidgetItem(str(self.e_data[i, j])))
            self.dlg.tableWidgetEndmember.setEditTriggers(QAbstractItemView.NoEditTriggers)

            for i in range(ncols):
                plt.plot(self.e_wlengths, self.e_data[:,i], label=self.e_names[i])
            plt.grid()
            plt.ylabel('Reflectance')
            if self.e_wlengths.dtype == 'float32' or self.e_wlengths.dtype == 'float64':
                plt.xlabel("Wavelength (Micrometers)")
            elif self.e_wlengths.dtype == 'int32' or self.e_wlengths.dtype == 'int64':
                plt.xlabel("Wavelength (Nanometers)")
            else:
                plt.xlabel("Wavelength")
            plt.legend(self.e_names)

            self.scene = QGraphicsScene(self.dlg)
            canvas = FigureCanvas(self.fig)

            self.scene.addWidget(canvas)
            self.dlg.graphicsViewSpectraPlot.setScene(self.scene)
            plt.close(self.fig)
            plt.rcParams.update(plt.rcParamsDefault)

    def loadendmember_check(self):
        """ Check if endmember values are loaded before processing """
        try:
            self.e_data
        except AttributeError:
            self.message = QMessageBox.warning(
                None, "Error", "No endmember spectra is loaded!", 
                QMessageBox.Ok, QMessageBox.Ok)
            return False


    def collinearity_check(self):
        """ Calculates variance inflation factor (VIF) in order to 
        detect multicollinearity in endmember spectra matrix

        Ref: Van der Meer, Freek D., and Xiuping Jia. "Collinearity 
        and orthogonality of endmembers in linear spectral unmixing.
        " International Journal of Applied Earth Observation and 
        Geoinformation 18 (2012): 491-503. """
        loadcheck = self.loadendmember_check()
        if (loadcheck) == False:
            return
        else:
            # Calculate Pearson product-moment correlation coefficients 
            # for endmember matrix columns (endmembers)

            # The correlation coefficients exceeding the value of 0.6 
            # indicates that the endmember matrix is tend to be ill-defined
            # and matrix inversion during linear spectral unmixing process 
            # is prone to errors.
            corr_coef = np.corrcoef(self.e_data, rowvar=False)
            # Get the upper-triangle of matrix corr_coef 
            # (Since corr_coef is symmetric) with diagonals-zeroed
            corr_coef_ut = np.triu(corr_coef, k=1)
            # Calculate VIF
            try:
                vif = np.linalg.inv(corr_coef).diagonal()
            except np.linalg.LinAlgError as err:
                if str(err) == 'Singular matrix':
                    self.message = QMessageBox.warning(
                    None, "Error", "Endmember matrix is singular. Please check your spectra.",
                    QMessageBox.Ok, QMessageBox.Ok)
                    return False
            # Check if endmembers with correlation coefficient higher than 0.6 and VIF > 10 exists
            if (np.any((np.logical_or(corr_coef_ut > 0.6,  corr_coef_ut < -0.6))) and 
                np.any((np.array(vif) > 10))):
                checkcollbox = QMessageBox()
                checkcollbox.setIcon(QMessageBox.Question)
                checkcollbox.setWindowTitle('Warning')
                checkcollbox.setText('Given endmember matrix does not met collinearity \
                    condition "Variance inflation factor > 10" Proceed? or Exit?')
                checkcollbox.setStandardButtons(
                    QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
                buttonYes = checkcollbox.button(
                    QMessageBox.StandardButton.Yes)
                buttonYes.setText('Proceed')
                buttonNo = checkcollbox.button(QMessageBox.StandardButton.No)
                buttonNo.setText('Exit')
                checkcollbox.exec_()
                if checkcollbox.clickedButton() == buttonNo:
                    return False
                elif checkcollbox.clickedButton() == buttonYes:
                    return True
            else:
                return False

    def start_unmixing_task(self):
        """ Start linear spectral unmixing """
    
        if self.dlg.mMapLayerComboBox.currentLayer() is None:
            self.message = QMessageBox.warning(
                None, "Error", "No image is loaded!", 
                QMessageBox.Ok, QMessageBox.Ok)
            return
        else:
            img_path = self.dlg.mMapLayerComboBox.currentLayer().source()
            ds_img = gdal.Open(img_path)
            if ds_img is None:
                self.message = QMessageBox.warning(
                    None, "Error", "Error reading image file!", 
                    QMessageBox.Ok, QMessageBox.Ok)
                return
        
        loadcheck = self.loadendmember_check()
        if (loadcheck) == False:
            return
        if self.dlg.radioButtonSCLS.isChecked() == True:
            constraint = "SCLS"
        elif self.dlg.radioButtonNNCLS.isChecked() == True:
            constraint = "NNCLS"
        else:
            constraint = None
        
        if self.dlg.lineEditOutput.text() == '':
            self.message = QMessageBox.warning(
                None, "Error", "No output directory is selected!", 
                QMessageBox.Ok, QMessageBox.Ok)
            return
        elif os.path.exists(self.dlg.lineEditOutput.text()) == False:
            self.message = QMessageBox.warning(
                None, "Error", "Output directory does not exists or permission error", 
                QMessageBox.Ok, QMessageBox.Ok)
            return
        elif os.path.exists(self.dlg.lineEditOutput.text()):
            outdir = self.dlg.lineEditOutput.text()

        if self.dlg.checkBoxErrorBand.isChecked():
            errorband = True
        elif not self.dlg.checkBoxErrorBand.isChecked():
            errorband = False

        collinearity = self.collinearity_check()
        if collinearity == False:
            return

        if (self.e_data.shape[0] <= self.e_data.shape[1]):
            self.message = QMessageBox.warning(None,
            "Error", " Invalid spectra matrix dimensions.",
            QMessageBox.Ok, QMessageBox.Ok)
            return

        if ((ds_img.RasterCount < self.e_data.shape[0]) or 
             ds_img.RasterCount > self.e_data.shape[0]):
            self.message = QMessageBox.warning(None,
            "Error", 
            "Number of input bands does not match endmember spectra row count. " +
            "Please check your input image/endmembers.",
            QMessageBox.Ok, QMessageBox.Ok)
            return

        unmix_algorithm = str(self.dlg.comboBoxAlgorithm.currentText())

        self.thread = QThread()
        self.unmixing_task = UnmixingTask(
            ds_img, 
            self.e_data, 
            self.e_names, 
            unmix_algorithm, 
            constraint,
            outdir,
            errorband)

        self.unmixing_task.moveToThread(self.thread)
        self.thread.started.connect(self.unmixing_task.run)
        self.unmixing_task.finished.connect(self.thread.quit)
        self.unmixing_task.finished.connect(self.unmixing_task.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.unmixing_task.prog.connect(self.show_progress)
        self.unmixing_task.proc.connect(self.show_procdetail)
        self.thread.start()
        self.dlg.pushButtonRun.setEnabled(False)
        self.thread.finished.connect(lambda: self.dlg.pushButtonRun.setEnabled(True))
    
    def stop_unmixing_task(self):
        """ Stop linear unmixing """

        try:
            self.unmixing_task.stop()
        except AttributeError:
            pass
        try:
            if self.thread.isRunning():
                self.thread.quit()
        except (RuntimeError, AttributeError) as error:
            pass

    def show_progress(self, valp):
        """ Show unmixing progress info in progressBar 
        
        :param valp: Percent of processed data information emitted by unmixing_task prog signal
        :type valp: int """

        self.dlg.progressBar.setValue(valp)
    
    def show_procdetail(self, valp):
        """ Show unmixing progress detail in QGIS console
        
        :param valp: Percent of processed data information emitted by unmixing_task proc signal
        :type valp: str """

        self.dlg.processinglabel.setText(valp)

    def toggle_parameters(self):
        """ Toggling function for unmixing algorithm comboBox """

        combo_text = str(self.dlg.comboBoxAlgorithm.currentText())
        if combo_text == "Partially Constrained Least Squares":
            self.dlg.radioButtonSCLS.setHidden(False)
            self.dlg.radioButtonNNCLS.setHidden(False)
        if combo_text == "Unconstrained Least Squares":
            self.dlg.radioButtonSCLS.setHidden(True)
            self.dlg.radioButtonNNCLS.setHidden(True)
        if combo_text == "Fully Constrained Least Squares":
            self.dlg.radioButtonSCLS.setHidden(True)
            self.dlg.radioButtonNNCLS.setHidden(True)
# -*- coding: utf-8 -*-
"""
/***************************************************************************
 LeastCostPath
                                 A QGIS plugin
 This Plugin used to find the least-cost path between to location on DEM Surface using anisotropic accumulated-cost surface and A* algorithm.
                              -------------------
        begin                : 2015-08-15
        git sha              : $Format:%H$
        copyright            : (C) 2015 by Achmad Faizal P S
        email                : faizalprbw@Gmail.com
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
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication
from PyQt4.QtGui import QAction, QIcon, QFileDialog
# Initialize Qt resources from file resources.py
import resources
# Import the code for the dialog
from LeastCostPath_dialog import LeastCostPathDialog
import os.path
# Import QGIS GUI and CORE
from qgis.core import *
from qgis.gui import *
# Import module for processing
import os, csv, time
from osgeo import ogr, gdal
from gdalconst import *
import numpy as np
import math as mt



class LeastCostPath:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
       
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'LeastCostPath_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = LeastCostPathDialog()

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Least Cost Path')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'LeastCostPath')
        self.toolbar.setObjectName(u'LeastCostPath')

        #Add method to call input and output function
        self.dlg.Input_dem.clear()
        self.dlg.Browse_dem.clicked.connect(self.select_input_dem)
        self.dlg.Input_cost.clear()
        self.dlg.Browse_cost.clicked.connect(self.select_input_cost)
        self.dlg.Input_slope_w.clear()
        self.dlg.Browse_slope_w.clicked.connect(self.select_input_weight)
        self.dlg.Output_raster.clear()
        self.dlg.Browse_raster.clicked.connect(self.select_output_raster)
        self.dlg.Output_vector.clear()
        self.dlg.Browse_vector.clicked.connect(self.select_output_vector)
        self.dlg.Output_acc.clear()
        self.dlg.Browse_acc.clicked.connect(self.select_output_acc)

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API."""
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('LeastCostPath', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar. """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/LeastCostPath/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'PATH Generate..'),
            callback=self.run,
            parent=self.iface.mainWindow())


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Least Cost Path'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    #Add Input and Output Methods 
    def select_input_dem(self):
        filename_dem = QFileDialog.getOpenFileName(self.dlg,"","","")
        self.dlg.Input_dem.setText(filename_dem)
        
    def select_input_cost(self):
        filename_cost = QFileDialog.getOpenFileName(self.dlg,"","","")
        self.dlg.Input_cost.setText(filename_cost)
    
    def select_input_weight(self):
        filename_slope_w = QFileDialog.getOpenFileName(self.dlg,"","","*.csv")
        self.dlg.Input_slope_w.setText(filename_slope_w)

    def select_output_raster(self):
        filename_raster = QFileDialog.getSaveFileName(self.dlg,"","","*.asc")
        self.dlg.Output_raster.setText(filename_raster)
    
    def select_output_vector(self):
        filename_vector = QFileDialog.getSaveFileName(self.dlg,"","","")
        self.dlg.Output_vector.setText(filename_vector)

    def select_output_acc(self):
        filename_acc = QFileDialog.getSaveFileName(self.dlg,"","","*.asc")
        self.dlg.Output_acc.setText(filename_acc)

        
    def run(self):
        """Run method that performs all the real work"""
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            
            def error(n,i):
                if i == 2 or i == 3 :
                    if n == "none" :
                        return
                if os.path.isfile(n) and os.access(n, os.R_OK) and os.stat(n).st_size != 0 :
                    return 
                else :
                    self.iface.messageBar().pushMessage("ERROR :"," Make sure the data that you input is correct and do not allow empty input line..!  If you don't want to use 'cost surface' or 'slope weight', write 'none' in those input line.", level=QgsMessageBar.CRITICAL, duration=10)
                    
            # input DEM Surface 
            filename_dem = self.dlg.Input_dem.text()
            # input COST Surface 
            filename_cost = self.dlg.Input_cost.text()  
            # Input Weight for Slope
            filename_w = self.dlg.Input_slope_w.text()
            
            # Error Handling 
            error(filename_dem,1)
            error(filename_cost,2)
            error(filename_w,3)
            
            # Get DEM information and Write as Array  
            dem_surface = gdal.Open( filename_dem, GA_ReadOnly )
            geotransform_dem = dem_surface.GetGeoTransform()
            DEM_Information = [geotransform_dem[1], geotransform_dem[0], geotransform_dem[3], dem_surface.GetProjection(), dem_surface.RasterYSize, dem_surface.RasterXSize] # Cell_Size, Origin_X, Origin_Y, CRS, Row, Col
            DEM_Value = np.array(dem_surface.GetRasterBand(1).ReadAsArray(), dtype ="float") #Raster to Array
            
            # Get COST information and Write as Array
            if filename_cost == "none" :
                COST_Value = np.zeros(shape=(dem_surface.RasterYSize,dem_surface.RasterXSize), dtype = "float") #Raster to Array
            else :
                cost_surface = gdal.Open( filename_cost, GA_ReadOnly )
                geotransform_cost = cost_surface.GetGeoTransform()
                COST_Information = [geotransform_cost[1], geotransform_cost[0], geotransform_cost[3], cost_surface.GetProjection(), cost_surface.RasterYSize, cost_surface.RasterXSize]  # Cell_Size, Origin_X, Origin_Y, CRS, Row, Col
                # Error Handling
                if DEM_Information == COST_Information :
                    COST_Value = np.array(cost_surface.GetRasterBand(1).ReadAsArray(), dtype ="float") #Raster to Array
                else : 
                    self.iface.messageBar().pushMessage("Cost surface's basic parameter (Cell Size, Origin, CRS, Resolution) must be equal with DEM surface..!", level=QgsMessageBar.CRITICAL, duration=10)
                    
            # Read Slope Weight CSV file as array
            if filename_w == "none" :
                W_CSV = np.array([[1,-90,90,1]], dtype = "float")
            else :
                W_CSV = np.genfromtxt(filename_w, delimiter=',')
              
            # Initialize Data Basic Parameter
            Col = dem_surface.RasterXSize
            Row = dem_surface.RasterYSize
            Origin_X = geotransform_dem[0]
            Origin_Y = geotransform_dem[3]
            Cell_Size = geotransform_dem[1]
            CRS = dem_surface.GetProjection()
            
            # Input Start and End Coordinate 
            start_X =  self.dlg.X_Start.text()
            start_Y =  self.dlg.Y_Start.text()
            end_X = self.dlg.X_End.text()
            end_Y =self.dlg.Y_End.text()
            
            # Error Handling
            if not start_X or not start_Y :
                self.iface.messageBar().pushMessage("ERROR : The Start Coordinate must be inputed..!!", level=QgsMessageBar.CRITICAL, duration=10) 
            if not end_X or not end_Y :
                self.iface.messageBar().pushMessage("ERROR : The End Coordinate must be inputed..!!", level=QgsMessageBar.CRITICAL, duration=10) 
            
            # Transformation from Cartesian to Raster system 
            sx = int(((float(start_X) - Origin_X)/Cell_Size)+1)
            sy = int(((Origin_Y - float(start_Y))/Cell_Size)+1)  
            start = sy,sx
            dx = int(((float(end_X) - Origin_X)/Cell_Size)+1)  
            dy = int(((Origin_Y - float(end_Y))/Cell_Size)+1) 
            end = dy,dx
            
            # Error Handling 
            if sx > Col or sx < 0 or sy > Row or sy < 0 :
                self.iface.messageBar().pushMessage("ERROR : The Inputed Coordinate is out of range", level=QgsMessageBar.CRITICAL, duration=10) 
            if dx > Col or dx < 0 or dy > Row or dy < 0 :
                self.iface.messageBar().pushMessage("ERROR : The Inputed Coordinate is out of range", level=QgsMessageBar.CRITICAL, duration=10) 
            
            #Heuristic Type
            ht = self.dlg.Heuristic.currentText()
            
            # Raster Output
            filename_ras = self.dlg.Output_raster.text()
            # Error Handling
            if not filename_ras :
                self.iface.messageBar().pushMessage("ERROR : The Output file must be inputed..!!", level=QgsMessageBar.CRITICAL, duration=10) 
                
            # Open and Write ESRI ASCII Header
            out_path = open(filename_ras, 'wb+')
            out_path.write('ncols         %i\n' % Col)
            out_path.write('nrows         %i\n' % Row)
            out_path.write('xllcorner     %f\n' % Origin_X)
            out_path.write('yllcorner     %f\n' % (Origin_Y - (Cell_Size * Row)) )
            out_path.write('cellsize      %i\n' % Cell_Size)
            out_path.write('NODATA_value  %i\n' % 0)
            
            # Vector Output
            filename = self.dlg.Output_vector.text()
            # Error Handling
            if not filename :
                self.iface.messageBar().pushMessage("ERROR : The Output file must be inputed..!!", level=QgsMessageBar.CRITICAL, duration=10)
                
            # Write as shapefile 
            filename_vec = filename +'.shp'
            filename_prj = filename +'.prj'
            fop = open(filename_prj, "w")
            fop.write(CRS)
            driver = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(filename_vec):
                driver.DeleteDataSource(filename_vec)
            data_source = driver.CreateDataSource(filename_vec)
            # Create attribute data table for shapefile 
            layer = data_source.CreateLayer(str(filename_vec), geom_type=ogr.wkbLineString )
            layer.CreateField(ogr.FieldDefn("Elev_1", ogr.OFTReal))
            layer.CreateField(ogr.FieldDefn("Elev_2", ogr.OFTReal))
            layer.CreateField(ogr.FieldDefn("Cost_1", ogr.OFTReal))
            layer.CreateField(ogr.FieldDefn("Cost_2", ogr.OFTReal))
            layer.CreateField(ogr.FieldDefn("Slope", ogr.OFTReal))
            layer.CreateField(ogr.FieldDefn("Distance", ogr.OFTReal))
            
            # Accumulated Cost Raster Output 
            filename_acc = self.dlg.Output_acc.text()
            # Error Handling
            if not filename_acc :
                self.iface.messageBar().pushMessage("ERROR : The Output file must be inputed..!!", level=QgsMessageBar.CRITICAL, duration=10)

            # Open and Write ESRI ASCII Header
            out_acc = open(filename_acc, 'wb+')
            out_acc.write('ncols         %i\n' % Col)
            out_acc.write('nrows         %i\n' % Row)
            out_acc.write('xllcorner     %f\n' % Origin_X)
            out_acc.write('yllcorner     %f\n' % (Origin_Y - (Cell_Size * Row)) )
            out_acc.write('cellsize      %i\n' % Cell_Size)
            out_acc.write('NODATA_value  %i\n' % 0)

            # Log File
            log_file = filename +'_log_data'+'.csv'
            log = open(log_file, "wb+")
            log_writer = csv.writer(log, delimiter=',')
            # Create header basic information  
            log_writer.writerow(['Number of Col :', Col])
            log_writer.writerow(['Number of Row :', Row])
            log_writer.writerow(['X- Axis Origin :', Origin_X])
            log_writer.writerow(['Y- Axis Origin :', Origin_Y])
            log_writer.writerow(['Spatial Resolution :', Cell_Size])
            log_writer.writerow(['Coordinate Reference System :', CRS])
            log_writer.writerow(['Start Col :', sx])
            log_writer.writerow(['Start Row :', sy])
            log_writer.writerow(['End Col :', dx])
            log_writer.writerow(['End Row :', dy])
            log_writer.writerow(['Heuristic Type :', ht])
            log_writer.writerow(['Current Row', 'Current Col','Current DEM Value','Current COST Value','No Candidate','Candidate Row','Candidate Col','Candidate DEM Value','Candidate COST Value','Slope (radian)','Slope Weight','Distance','Accumulated Cost', 'Heuristic', 'G Value', 'F Value' ])
            
            # Initialize A* Variable 
            open_set =set()
            close_set = set()
            parent = { }
            accumulated_cost = np.zeros(shape=(Row,Col), dtype = "float")
            
            #Anisotropic Cost Function
            def Anisotropic_Cost(j,k,y,x) :
                #Calculate Slope And Distance 
                if j == y or k == x : #Perpendicular Movement 
                    Slope = mt.atan( (DEM_Value[j-1][k-1] - DEM_Value[y-1][x-1])/Cell_Size ) 
                    Distance = mt.sqrt( Cell_Size**2 + ( DEM_Value[j-1][k-1] - DEM_Value[y-1][x-1] )**2 ) 
                    
                else : # Diagonal Movement
                    Slope = mt.atan( (DEM_Value[j-1][k-1] - DEM_Value[y-1][x-1])/((2**0.5)*Cell_Size) ) 
                    Distance = mt.sqrt( 2 * Cell_Size**2 + ( DEM_Value[j-1][k-1] - DEM_Value[y-1][x-1] )**2 )
                
                #Weight Identification
                for i in W_CSV : 
                    if Slope > mt.radians(i[1]) and Slope <= mt.radians(i[2]) :
                        Weight = i[3]
                        break
                    else :
                        Weight = 1 
                    
                #Calculate Anisotropic Cost
                G =  ( ( COST_Value[j-1][k-1] + COST_Value[y-1][x-1] )/2. + abs(Slope * Weight) ) * Distance    
                return [G, Slope, Weight, Distance]
            
            def Adjacent_Cell(y,x):
                candidate = []
                if y > 0: 
                    candidate.append((y-1, x))
                if y < Row : 
                    candidate.append((y+1, x)) 
                if x > 0: 
                    candidate.append((y, x-1)) 
                if x < Col : 
                    candidate.append((y, x+1)) 
                if x > 0 and y > 0:
                    candidate.append((y-1, x-1)) 
                if y < Row and x <  Col :
                    candidate.append((y+1, x+1)) 
                if y < Row and x > 0:
                    candidate.append((y+1, x-1))  
                if y > 0 and x <  Col :
                    candidate.append((y-1, x+1))
                return candidate
            
            def Heuristic(j,k):
                if ht == "A* Manhattan" :
                    h = ( abs(dx - k) + abs(dy - j) ) 
                elif ht == "A* Diagonal" :
                    xDis = abs(dx - k)
                    yDis = abs(dy - j)
                    if xDis > yDis :
                        h = (2**0.5 * yDis) + (xDis - yDis)
                    else :
                        h = (2**0.5 * xDis) + (yDis - xDis)
                elif ht == "A* Euclidean" :
                    h = (( abs(dx - k)**2 + abs(dy - j)**2 )**0.5) 
                return h/((Col+Row)/2.)*Cell_Size**2
            
            #A-star Function
            def A_star():
                # Message Bar
                self.iface.mainWindow().statusBar().showMessage("Please Wait..!  Program is being performed PATH search process..." )
                # Add start to Open Set, add start and its value from acc_c matrix  
                open_set.add(start)
                Open_value = ({start : accumulated_cost[start[0]][start[1]]} )
                # A* Condition
                while end not in open_set :
                    # Current Cell is the minimum value from open value, add it to close set and remove from open set 
                    current_cell = min(Open_value, key = Open_value.get)
                    close_set.add(current_cell)
                    open_set.remove(current_cell)
                    del Open_value[current_cell]
                    # Call 8 adjacent cell
                    y = current_cell[0]
                    x = current_cell[1]
                    candidate = Adjacent_Cell(y,x)
                    # A* candidate checking
                    n = 0 
                    for item in candidate :
                        j = item[0]
                        k = item[1]
                        n += 1
                        # Condition to be ignore 
                        if item in close_set or DEM_Value[j-1][k-1] == 0  :         
                            continue
                        # Condition to be identification 
                        else :
                            #Heuristic Type Identification   
                            h = Heuristic(j,k)
                            #Calculate COST
                            CAL = Anisotropic_Cost(j,k,y,x)
                            g = CAL[0]  
                            f = accumulated_cost[y-1][x-1] + g + h 
                            # Condition where candidate is not in open set
                            if item not in open_set :     
                                open_set.add(item)
                                parent.update({item : current_cell})
                                accumulated_cost[j-1][k-1] = f
                                Open_value.update( {item : f } )
                            # Condition where candidate already in open set
                            else :
                                if f < accumulated_cost[j-1][k-1] :
                                    parent.update({item : current_cell})
                                    accumulated_cost[j-1][k-1] = f
                                    Open_value.update( {item : f } )
                        # Write Log            
                        log_row = [y,x,DEM_Value[y-1][x-1],COST_Value[y-1][x-1],n,j,k,DEM_Value[j-1][k-1],COST_Value[j-1][k-1],CAL[1],CAL[2],CAL[3],accumulated_cost[y-1][x-1],h,g,f]
                        log_writer.writerow(log_row)    
                # Generate PATH from parent by back link processes    
                def root_path(parent, end) : 
                        yield end
                        while end in parent:
                            end = parent[end]
                            yield end    
                                 
                PATH_ID =  list(root_path(parent, end))
                PATH_List = []
                log_writer.writerow(['PATH ROW', 'PATH COL'])  
                
                OutRas = np.zeros(shape=(Row,Col),dtype = int)
                for i in PATH_ID :
                    r = i[0]
                    c = i[1]
                    PATH_List.append((c-1,r-1))
                    OutRas[r-1][c-1] = 1 
                    log_writer.writerow([r-1,c-1])  
                    
                PATH_RASTER = np.savetxt(out_path, OutRas, fmt="%4i") 
                COST_RASTER = np.savetxt(out_acc, accumulated_cost, fmt="%4i") 
                    
                j = 0
                while j < len(PATH_List) - 1 :
                    OutVec = ogr.Geometry(ogr.wkbLineString)
                    feature = ogr.Feature(layer.GetLayerDefn())
                    X1 = PATH_List[j][0] 
                    Y1 = PATH_List[j][1] 
                    X2 = PATH_List[j+1][0] 
                    Y2 = PATH_List[j+1][1]
                    OutVec.AddPoint(Origin_X + (Cell_Size*X1) + Cell_Size/2. ,Origin_Y - (Cell_Size*Y1) - Cell_Size/2. )
                    OutVec.AddPoint(Origin_X + (Cell_Size*X2) + Cell_Size/2. ,Origin_Y - (Cell_Size*Y2) - Cell_Size/2. )
                    feature.SetField("Elev_1", DEM_Value[Y1][X1])
                    feature.SetField("Elev_2", DEM_Value[Y2][X2])
                    feature.SetField("Cost_1", COST_Value[Y1][X1])
                    feature.SetField("Cost_2", COST_Value[Y2][X2])
                    if Y2 == Y1 or X2 == X1 :
                        slope_ang =  mt.degrees( mt.atan( (DEM_Value[Y2][X2] - DEM_Value[Y1][X1])/Cell_Size ) )
                        slope_dis = mt.sqrt( Cell_Size**2 + ( DEM_Value[Y2][X2] - DEM_Value[Y1][X1] )**2 ) 
                    else : 
                        slope_ang =  mt.degrees( mt.atan( (DEM_Value[Y2][X2] - DEM_Value[Y1][X1])/( (2**0.5)*Cell_Size) ) )
                        slope_dis = mt.sqrt( 2 * Cell_Size**2 + ( DEM_Value[Y2][X2] - DEM_Value[Y1][X1] )**2 )
                    feature.SetField("Slope", slope_ang)
                    feature.SetField("Distance", slope_dis)
                    feature.SetGeometry(OutVec)
                    PATH_VECTOR = layer.CreateFeature(feature)
                    j += 1
                    
                return PATH_RASTER, PATH_VECTOR, COST_RASTER
            
            
            # lets run ...!!!
            start_time = time.time() #start time for evaluate
            
            A_star()
            
            self.iface.addRasterLayer(filename_ras, "PATH_Raster")
            self.iface.addVectorLayer(filename_vec, "PATH_Vector", "ogr")
            
            self.iface.mainWindow().statusBar().showMessage(("--- %s seconds ---" % (time.time() - start_time)) )
            self.iface.messageBar().clearWidgets()
            pass
